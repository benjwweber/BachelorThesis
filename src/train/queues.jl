ENV["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
using Lux
using Enzyme
using Reactant
using Optimisers
using Random
rng = Random.Xoshiro(42)
include("./src/data/Data.jl")
using .Data
include("./src/models/Models.jl")
using .Models
include("./src/losses/Losses.jl")
using .Losses

using Functors
using Accessors
using Transducers
using JLD2
using MLUtils
using BFloat16s

include("./src/train/schedulers.jl")
include("./src/train/optimiser.jl")
include("./src/train/log.jl")
include("./src/train/clip.jl")
include("./src/data/load/fix.jl")

# Devices
const cpu = cpu_device()
const gpus = Reactant.devices()
const ngpus = length(gpus)

const mesh = Reactant.Sharding.Mesh(
    gpus,
    (:batch,)
)
const device_replicated = reactant_device(
    sharding=Reactant.Sharding.Replicated(mesh),
    track_numbers=AbstractFloat
)
const device_sharded = reactant_device(
    BFloat16;
    sharding=Reactant.Sharding.DimsSharding(mesh, (-1,), (:batch,)),
)

# DataLoader
paths = load("./data/paths.jld2", "paths");
tiles = paths |> MapCat(Data.gettiles) |> tcollect;
tileloader = TileLoader(tiles);
dinoset = DINOSet(
    tileloader;
    dims_global=(256, 256),
    scale_global=(0.32, 1.0),
    nviews_global=2,
    dims_local=(128, 128),
    scale_local=(0.05, 0.32),
    nviews_local=8,
    dims_mask=(256, 256) .÷ 16,
    probability_mask=0.5,
    npatches_min=0.1,
    npatches_max=0.5,
    area_min=4,
    area_max=64,
    ratio_min=0.3,
    ratio_max=1/0.3
);

batchsize = 16 * ngpus
dataloader = DataLoader(
    dinoset;
    batchsize,
    shuffle=false,
    parallel=false, #true,
    num_workers=16,
    partial=false,
    collate=true
);

# Models
model_student = Models.MixedPrecision{BFloat16,Float32}(
    Models.Student(
        Models.vit_large(;
            kwargs_blocks=(; probability_dropout_path=0.3f0)
        ),
        Models.DINOHead(1024 => 2048 => 256 => 4096),
        Models.DINOHead(1024 => 2048 => 256 => 4096),
    );
);
model_teacher = Models.MixedPrecision{BFloat16,Float32}(
    Models.Teacher(
        Models.vit_large(;
        # kwargs_blocks=(; probability_dropout_path=0.3f0)
        ),
        Models.DINOHead(1024 => 2048 => 256 => 4096),
        Models.DINOHead(1024 => 2048 => 256 => 4096),
    );
);
models = (;
    student=model_student,
    teacher=model_teacher
);
parameters, states = load(
    "./data/checkpoints/0.jld2",
    "parameters",
    "states"
);

teacher = LuxCore.StatefulLuxLayer(
    models.teacher,
    parameters.teacher |> device_replicated,
    states.teacher |> LuxCore.testmode |> bf16 |> device_replicated
);

# Compiles
batch = first(dataloader);
GC.gc(true)
inputs = batch |> device_sharded;
GC.gc(true)
teacher_compiled = @compile Models.compute_outputs(teacher, inputs);

nbatches = 16
dino = []
ibot = []
mask = []
GC.gc(true)
for (i, batch) in enumerate(dataloader)
    inputs = batch |> device_sharded
    outputs = teacher_compiled(teacher, inputs)
    GC.gc(true)
    push!(dino, cpu(outputs.dino))
    push!(ibot, cpu(outputs.ibot))
    push!(mask, Models.WHCVB_to_CWxHVB(cpu(inputs.mask)))
    i >= nbatches && break
end
GC.gc(true)
queues = (;
    dino=reduce(Models.bcat, dino),
    ibot=reduce(Models.bcat, ibot),
    mask=reduce(Models.bcat, mask)
);

jldsave("./data/queues/1024.jld2"; queues)
