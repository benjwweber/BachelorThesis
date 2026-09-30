ENV["XLA_REACTANT_GPU_PREALLOCATE"] = "false"
ENV["XLA_REACTANT_GPU_MEM_FRACTION"] = "0.95"
ENV["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
using Logging
using LoggingExtras
using Dates
s = string(now())
checkpoints = mkpath("./data/checkpoints/" * s)
logs = mkpath("./logs/" * s)
logs_losses = touch(logs * "/losses.log")
logs_stderr = open(touch(logs * "/stderr.log"), "a")
logger = TeeLogger(
    ConsoleLogger(stdout),
    FileLogger(logs * "/logs.log", append=true, always_flush=true)
)
global_logger(logger)
redirect_stderr(logs_stderr)
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
include("./src/train/clipnorm.jl")
include("./src/data/fix.jl")


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

# Schedules
epochs = 256 # 256 <- maybe smaller
nepochs = 1000 # 1000
niterations = epochs * nepochs
saveiteration = 256
schedules = (;
    etas=(;
        backbone=TriangleCosAnnealConstant(;
            start=0.0f0, peak=5.0f-5, stop=1.0f-6,
            niterations=(;
                freeze=1epochs,
                warmup=10epochs,
                cosine=niterations - 11epochs,
            ),
            eltype=Float32,
            device=device_replicated
        ),
        heads=TriangleCosAnnealConstant(;
            start=0.0f0, peak=0.001f0, stop=1.0f-6,
            niterations=(;
                freeze=0,
                warmup=3epochs,
                cosine=niterations - 3epochs,
            ),
            eltype=Float32,
            device=device_replicated
        ),
        lasts=TriangleCosAnnealConstant(;
            start=0.0f0, peak=0.001f0, stop=1.0f-6,
            niterations=(;
                freeze=1epochs,
                warmup=2epochs,
                cosine=niterations - 3epochs,
            ),
            eltype=Float32,
            device=device_replicated
        )
    ),
    # lambda=DeviceScheduler{Float32}(Returns(0.4f0), device_replicated),
    lambda=TriangleCosAnnealConstant(;
        start=0.04f0, peak=0.04f0, stop=0.4f0,
        niterations=(;
            freeze=0,
            warmup=0,
            cosine=niterations,
        ),
        eltype=Float32,
        device=device_replicated
    ),
    momentum=TriangleCosAnnealConstant(;
        start=0.992f0, peak=0.992f0, stop=1.0f0,
        niterations=(;
            freeze=0,
            warmup=0,
            cosine=niterations,
        ),
        eltype=Float32,
        device=device_replicated
    ),
    temperature=TriangleCosAnnealConstant(;
        start=0.02f0, peak=0.07f0, stop=0.07f0,
        # freeze = 0.04f0
        niterations=(;
            freeze=0, # freeze = 10epochs,
            warmup=30epochs,
            cosine=niterations - 30epochs,
        ),
        eltype=Float32,
        device=device_replicated
    )
);

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

batchsize = 16 * ngpus # 64 ??
dataloader = DataLoader(
    dinoset;
    batchsize,
    shuffle=true,
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
model_teacher = Models.MixedPrecision{BFloat16,BFloat16}(
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
    "./data/checkpoints/dinov3_vitl16_pretrain_lvd168m.jld2",
    "parameters",
    "states"
);

# Optimiser
omega = 3.0f0 # not on device!
naccumulate = 16
optimiser = AdamW(;
    eta=0.0f0, #1.0f-6,
    beta=(0.9f0, 0.999f0),
    lambda=0.04f0,
    epsilon=1.0f-8
);
factors = (;
    etas=compute_factors_eta(parameters.student, 0.9f0; embedding_patch=0.2f0),
    lambda=compute_factors_lambda(parameters.student, 1.0f0; head_dino=1.0f0)
);

# Student and Teacher
states_objective = (; weights=(;
    dino=(; var"global"=1.0f0, var"local"=1.0f0),
    ibot=1.0f0,
    koleo=0.1f0
));
student = Lux.Training.TrainState(
    models.student,
    parameters.student |> device_replicated,
    (; student=states.student, objective=states_objective) |> bf16 |> device_replicated,
    optimiser
);
teacher = LuxCore.StatefulLuxLayer(
    models.teacher,
    parameters.teacher |> device_replicated,
    states.teacher |> LuxCore.testmode |> bf16 |> device_replicated
);
accumulator = parameters.student |> deepcopy |> device_replicated;

# Automatic Differentiation
autodiff = AutoReactant();

# Objective
objective = Losses.PreTrainingObjective(
    Losses.DINOLoss(; temperature=0.1f0, ignore_diagonal=Val(true)),
    Losses.DINOLoss(; temperature=0.1f0, ignore_diagonal=Val(false)),
    Losses.iBOTLoss(; temperature=0.1f0),
    Losses.KoLeoLoss(; epsilon=1f-8)
);

# Compiles

iteration = 1
step = 1
inputs = first(dataloader) |> device_sharded;
etas=(;
    backbone=schedules.etas.backbone(step),
    heads=schedules.etas.heads(step),
    lasts=schedules.etas.lasts(step),
);
lambda=schedules.lambda(step);
temperature=schedules.temperature(step);
momentum=schedules.momentum(step);
GC.gc(true)
queues = load(
    "./data/queues/1024.jld2",
    "queues"
) |> device_sharded;
compute_etas_lambdas_compiled = @compile compute_etas_lambdas(
    etas,
    factors.etas,
    lambda,
    factors.lambda
);
student = Optimisers.adjust!(student; compute_etas_lambdas_compiled(
    etas,
    factors.etas,
    lambda,
    factors.lambda
)...);
etas = nothing;
lambda = nothing;
GC.gc(true)
update_queues_compiled!, queues = Reactant.with_profiler("./profile/update_queues") do
    global queues
    update_queues_compiled! = Reactant.Profiler.annotate("compile") do
        global queues
        global inputs
        global teacher
        @compile Models.update_queues!(teacher, inputs, queues)
    end
    queues = Reactant.Profiler.annotate("first") do
        global queues
        global inputs
        global teacher
        update_queues_compiled!(teacher, inputs, queues)
    end
    return update_queues_compiled!, queues
end;
GC.gc(true)
compute_targets_compiled, targets = Reactant.with_profiler("./profile/compute_targets") do
    global queues
    global temperature
    global batchsize
    compute_targets_compiled = Reactant.Profiler.annotate("compile") do
        global queues
        global temperature
        global batchsize
        @compile Models.compute_targets(queues, temperature, batchsize)
    end
    targets = Reactant.Profiler.annotate("first") do
        global queues
        global temperature
        global batchsize
        compute_targets_compiled(queues, temperature, batchsize)
    end
    return compute_targets_compiled, targets
end;
temperature = nothing;
GC.gc(true)
gradients, loss, statistics, student = Reactant.with_profiler("./profile/compute_gradients") do
    Training.compute_gradients(
        autodiff,
        objective,
        (; inputs, targets),
        student,
    )
end;
targets = nothing;
inputs = nothing;
GC.gc(true)
decumulate_compiled!, accumulator = Reactant.with_profiler("./profile/decumulate_gradients") do
    global accumulator
    decumulate_compiled! =  Reactant.Profiler.annotate("compile") do
        global accumulator
        @compile decumulate!(accumulator)
    end
    accumulator =  Reactant.Profiler.annotate("first") do
        global accumulator
        decumulate_compiled!(accumulator)
    end
    return decumulate_compiled!, accumulator
end;
GC.gc(true)
accumulate_compiled!, accumulator = Reactant.with_profiler("./profile/accumulate_gradients") do
    global accumulator
    global gradients
    global naccumulate
    accumulate_compiled! =  Reactant.Profiler.annotate("compile") do
        global accumulator
        global gradients
        global naccumulate
        @compile accumulate!(accumulator, gradients, naccumulate)
    end
    accumulator =  Reactant.Profiler.annotate("first") do
        global accumulator
        global gradients
        global naccumulate
        accumulate_compiled!(accumulator, gradients, naccumulate)
    end
    return accumulate_compiled!, accumulator
end;
gradients = nothing;
GC.gc(true)
clipnorm_backbone_heads_compiled = Reactant.with_profiler("./profile/clipnorm_gradients") do
    global accumulator
    global omega
    clipnorm_backbone_heads_compiled = Reactant.Profiler.annotate("compile") do
        global accumulator
        global omega
        @compile clipnorm_backbone_heads(accumulator, omega)
    end
    Reactant.Profiler.annotate("first") do
        global accumulator
        global omega
        clipnorm_backbone_heads_compiled(accumulator, omega)
    end
    return clipnorm_backbone_heads_compiled
end;
GC.gc(true)
t0 = time();
@info "starting apply_gradients compile" now()
student = Reactant.with_profiler("./profile/apply_gradients") do
    global student
    global accumulator
    global omega
    Training.apply_gradients!(
        student,
        clipnorm_backbone_heads_compiled(accumulator, omega)
    )
end;
t1 = time()
@show (t1 - t0)/60/60
update_teacher_compiled, teacher = Reactant.with_profiler("./profile/update_teacher") do
    global teacher
    global student
    global momentum
    update_teacher_compiled = Reactant.Profiler.annotate("compile") do
        global teacher
        global student
        global momentum
        @compile Models.update_teacher(teacher.ps, student.parameters, momentum)
    end
    teacher.ps = Reactant.Profiler.annotate("first") do
        global teacher
        global student
        global momentum
        update_teacher_compiled(teacher.ps, student.parameters, momentum)
    end
    return update_teacher_compiled, teacher
end;
log_pretraining(logs_losses, step, iteration, cpu(loss), cpu(statistics));

inputs = nothing;
etas = nothing;
lambda = nothing;
temperature = nothing;
momentum = nothing;
targets = nothing;
gradients = nothing;
loss = nothing;
statistics = nothing;
# clipped = nothing;
GC.gc(true);
GC.gc(false);
deviceloader = Channel(1; spawn=true) do channel
    for (iteration, batch) in enumerate(dataloader)
        step = Int(ceil(iteration / naccumulate))
        data = (;
            iteration,
            step,
            inputs=batch |> device_sharded,
            etas=(;
                backbone=schedules.etas.backbone(step),
                heads=schedules.etas.heads(step),
                lasts=schedules.etas.lasts(step),
            ),
            lambda=schedules.lambda(step),
            temperature=schedules.temperature(step),
            momentum=schedules.momentum(step)
        )
        put!(channel, data)
    end
end
GC.gc(true);

student, teacher, queues, accumulator = Reactant.with_profiler("./profile/loop") do
    global student
    global teacher
    global queues
    global accumulator
    for (; iteration, step, inputs, etas, lambda, temperature, momentum) in deviceloader
        if step > 5
            break
        end
        @show step, iteration
        if iteration % naccumulate == 0
            student = Optimisers.adjust!(student; compute_etas_lambdas_compiled(
                (;
                    backbone=etas.backbone,
                    heads=etas.heads,
                    lasts=etas.lasts,
                ),
                factors.etas,
                lambda,
                factors.lambda
            )...)
        end
        queues = update_queues_compiled!(teacher, inputs, queues)
        targets = compute_targets_compiled(queues, temperature, batchsize)
        gradients, loss, statistics, student = Training.compute_gradients(
            autodiff,
            objective,
            (; inputs, targets),
            student
        )
        # free!(inputs)
        accumulator = accumulate_compiled!(accumulator, gradients, naccumulate)
        if iteration % naccumulate == 0
            student = Training.apply_gradients!(
                student,
                clipnorm_backbone_heads_compiled(accumulator, omega)
            )
            teacher.ps = update_teacher_compiled(teacher.ps, student.parameters, momentum)
            accumulator = decumulate_compiled!(accumulator)
            GC.gc(false)
        end
        # free!(gradients)

        log_pretraining(logs_losses, step, iteration, cpu(loss), cpu(statistics))
        # free!(loss)
        # free!(statistics)
        # free!(etas)...
        # GC.gc(false)
        if step % 16 == 0 && iteration % naccumulate == 0
            jldsave(
                "$checkpoints/$iteration.jld2";
                parameters=(;
                    student=cpu(student.parameters),
                    teacher=cpu(teacher.ps)
                ),
                states=(;
                    student=cpu(student.states),
                    teacher=cpu(teacher.st)
                ),
            )
            GC.gc(true)
        end
    end
    return student, teacher, queues, accumulator
end;

# function checkpoint(teacher, student, step, checkpoints)
#     jldsave(
#         "$checkpoints/$step.jld2";
#         parameters=(;
#             student=cpu(student.parameters),
#             teacher=cpu(teacher.ps)
#         ),
#         states=(;
#             student=cpu(student.states),
#             teacher=cpu(teacher.st)
#         ),
#     )
# end

# function train(
#     teacher, student, queues, accumulator, dataloader;
#     naccumulate, factors, batchsize, autodiff, objective, omega, logs_losses, checkpoints
# )
#     deviceloader = Channel(1; spawn=true) do channel
#         for (iteration, batch) in enumerate(dataloader)
#             step = Int(ceil(iteration / naccumulate))
#             data = (;
#                 iteration,
#                 step,
#                 inputs=batch |> device_sharded,
#                 etas=(;
#                     backbone=schedules.etas.backbone(step),
#                     heads=schedules.etas.heads(step),
#                     lasts=schedules.etas.lasts(step),
#                 ),
#                 lambda=schedules.lambda(step),
#                 temperature=schedules.temperature(step),
#                 momentum=schedules.momentum(step)
#             )
#             put!(channel, data)
#         end
#     end
#     for (; iteration, step, inputs, etas, lambda, temperature, momentum) in deviceloader
#         @show step, iteration
#         if iteration % naccumulate == 0
#             student = Optimisers.adjust!(student; compute_etas_lambdas_compiled(
#                 (;
#                     backbone=etas.backbone,
#                     heads=etas.heads,
#                     lasts=etas.lasts,
#                 ),
#                 factors.etas,
#                 lambda,
#                 factors.lambda
#             )...)
#         end
#         queues = update_queues_compiled!(teacher, inputs, queues)
#         targets = compute_targets_compiled(queues, temperature, batchsize)
#         current_student = student
#         gradients, loss, statistics, student = Training.compute_gradients(
#             autodiff,
#             objective,
#             (; inputs, targets),
#             current_student
#         )
#         accumulator = accumulate_compiled!(accumulator, gradients, naccumulate)
#         if iteration % naccumulate == 0
#             student = Training.apply_gradients!(
#                 student,
#                 clipnorm_backbone_heads_compiled(accumulator, omega)
#             )
#             teacher.ps = update_teacher_compiled(teacher.ps, student.parameters, momentum)
#             accumulator = decumulate_compiled!(accumulator)
#             GC.gc(false)
#         end
#
#         log_pretraining(logs_losses, step, iteration, cpu(loss), cpu(statistics))
#         if step % 16 == 0 && iteration % naccumulate == 0
#             checkpoint(teacher, student, step, checkpoints)
#             GC.gc(true)
#         end
#     end
#     return teacher, student, queues, accumulator
# end
