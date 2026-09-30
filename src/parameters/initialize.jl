using Lux
using Random
using JLD2
using Accessors
using BFloat16s
using Functors
using NamedTupleTools
include("./src/models/Models.jl")
using .Models

model = Models.MixedPrecision{BFloat16,Float32}(
    Models.Student(
        Models.vit_large(;
            kwargs_blocks=(; probability_dropout_path=0.3f0)
        ),
        DINOHead(1024 => 2048 => 256 => 4096), # 65536),
        DINOHead(1024 => 2048 => 256 => 4096), # 4096),
    )
);

rng = Random.Xoshiro(42);
parameters, states = LuxCore.setup(rng, model);

parameters_backbone = load("./data/parameters/lux/dinov3_vitl16_pretrain_lvd168m.jld2", "parameters");

states = @modify x->Float32.(BFloat16.(x)) states.backbone.embedding_rotarypositional.periods;

@reset parameters.backbone = merge_recursive(
    parameters.backbone, parameters_backbone
);
parameters = (;
    student=parameters,
    teacher=deepcopy(parameters)
);
states = (;
    student=states,
    teacher=LuxCore.testmode(deepcopy(states))
);
jldsave(
    "./data/checkpoints/dinov3_vitl16_pretrain_lvd168m.jld2";
    parameters,
    states
);
