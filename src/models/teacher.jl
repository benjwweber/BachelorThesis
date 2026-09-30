using Functors
using Accessors
using Reactant

@containerlayer struct Teacher
    backbone
    head_dino
    head_ibot
end

@apply function (teacher::Teacher)(
    inputs #(; views = (; global, local), masks = (; global, local))
)
    nprefix = teacher.backbone.token_prefix.dims[2]
    nbatches = size(inputs.views.global, 5)
    nviews = size(inputs.views.global, 4)

    output = teacher.backbone(flatten_last(inputs.views.global, 2), nothing)

    class = @view output[:, 1, :]
    patches = @view output[:, (nprefix+1):end, :]

    return (;
        dino=reshape_last(teacher.head_dino(class), nviews, nbatches),
        ibot=reshape_last(teacher.head_ibot(patches), nviews, nbatches),
        mask=WHCVB_to_CWxHVB(inputs.mask)
    )
end

function WHCVB_to_CWxHVB(
    x # |> size == (W, H, C, V, B)
)
    return permutedims(
        reshape(
            x,
            (:, size(x, 3), size(x, 4), size(x, 5))
        ), # |> size == (W×H, C, B)
        (2, 1, 3, 4)
    ) # |> size == (C, W×H, B)
end

function compute_targets(teacher, inputs, temperature)
    outputs = teacher(inputs)
    targets = (;
        dino=sinkhornknopp_logspace(outputs.dino, temperature), # maybe only sinkhornknopp
        ibot=sinkhornknopp_logspace(outputs.ibot, WHCVB_to_CWxHVB(inputs.mask), temperature),
    )
    return targets
end

function getbatch(x, indices)
    return x[ntuple(Returns(:), ndims(x)-1)..., indices]
end

function bcat(arg, args...)
    return cat(arg, args..., dims=ndims(arg))
end

function compute_outputs(teacher, inputs)
    return teacher(inputs)
end
# function compute_targets(teacher, inputs, queues, temperature)
#     nqueued = size(queues.dino, ndims(queues.dino))
#     nbatch = size(queues.dino, ndims(queues.dino))
#     outputs = teacher(inputs)
#     queues.dino[]
# end

function compute_targets!(teacher, inputs, queues, temperature)
    n = size(inputs.views.global, ndims(inputs.views.global))
    N = size(queues.dino, ndims(queues.dino))
    outputs = teacher(inputs)
    batchindices = axes(outputs.dino, ndims(outputs.dino))
    queueindices = axes(queues.dino, ndims(queues.dino))

    temp_dino = bcat(outputs.dino, queues.dino)
    temp_ibot = bcat(outputs.ibot, queues.ibot)
    temp_mask = bcat(outputs.mask, queues.mask)

    targets = (;
        dino=getbatch(sinkhornknopp_logspace(temp_dino, temperature; niterations=3), batchindices),
        ibot=getbatch(sinkhornknopp_logspace(temp_ibot, temp_mask, temperature; niterations=3), batchindices)
    )

    queues.dino .= getbatch(temp_dino, queueindices)
    queues.ibot .= getbatch(temp_ibot, queueindices)
    queues.mask .= getbatch(temp_mask, queueindices)
    return targets, queues
end

function uff(teacher, inputs, queues, temperature)
    outputs = teacher(inputs)
    n = size(outputs.dino, ndims(outputs.dino))
    queues.dino[:, :, (begin+n):end] .= queues.dino[:, :, begin:(end-n)]
    queues.dino[:, :, begin:n] .= outputs.dino

    queues.ibot[:, :, :, (begin+n):end] .= queues.ibot[:, :, :, begin:(end-n)]
    queues.ibot[:, :, :, begin:n] .= outputs.ibot

    queues.mask[:, :, :, (begin+n):end] = queues.mask[:, :, :, 1:(end-n)]
    queues.mask[:, :, :, begin:n] .= outputs.mask

    targets = (;
        dino=sinkhornknopp_logspace(queues.dino, temperature; niterations=3)[:, :, begin:n],
        ibot=sinkhornknopp_logspace(queues.ibot, queues.mask, temperature; niterations=3)[:, :, :, begin:n],
    )
    return targets, queues
end

function update_queues!(teacher, inputs, queues)
    outputs = teacher(inputs)
    n = size(outputs.dino, ndims(outputs.dino))
    queues.dino[:, :, (begin+n):end] .= queues.dino[:, :, begin:(end-n)]
    queues.dino[:, :, begin:n] .= outputs.dino

    queues.ibot[:, :, :, (begin+n):end] .= queues.ibot[:, :, :, begin:(end-n)]
    queues.ibot[:, :, :, begin:n] .= outputs.ibot

    queues.mask[:, :, :, (begin+n):end] = queues.mask[:, :, :, 1:(end-n)]
    queues.mask[:, :, :, begin:n] .= outputs.mask

    return queues
end
function compute_targets(queues, temperature, n)
    return (;
        dino=sinkhornknopp_logspace(queues.dino, temperature)[:, :, begin:n],
        ibot=sinkhornknopp_logspace(queues.ibot, queues.mask, temperature)[:, :, :, begin:n],
    )
end

function ema(x, s, α)
    return (x * α) + (s * (1 - α))
end

function ahh(parameters_teacher, parameters_student, momentum)
    return fmap(
        Base.Fix{3}(ema, momentum),
        parameters_teacher,
        parameters_student
    )
end

function update_teacher!(teacher, student, momentum)
    fmap(teacher.ps, student.parameters) do t, s
        @. t = (t * momentum) + (s * (1 - momentum))
    end
    return nothing
end

# function update_teacher!(teacher, student, momentum)
#     return teacher.ps = fmap(
#         Base.Fix{3}(ema, momentum),
#         teacher.ps,
#         student.parameters
#     )
# end
