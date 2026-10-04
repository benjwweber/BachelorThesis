struct PreTrainingObjective
    dino_global
    dino_local
    ibot
    koleo
end

function (objective::PreTrainingObjective)(
    student,
    parameters,
    states, # (; student, objective)
    data #(; inputs = (; views = (; global, local), mask), targets = (; dino, ibot))
)
    outputs, states_student = student(data.inputs, parameters, states.student)

    nviews_global = size(data.inputs.views.global, 4)
    nviews_local = size(data.inputs.views.local, 4)

    terms_global = nviews_global * (nviews_global - 1) # nviews_global^2 # if ignore_diagonal == false
    terms_local = nviews_global * nviews_local

    T = Lux.unwrapped_eltype(outputs)
    scale_global = convert(T, (terms_global / (terms_global + terms_local)))
    scale_local = convert(T, (terms_local / (terms_global + terms_local)))

    losses = (;
        dino=(;
            var"global"=objective.dino_global(outputs.dino.global, data.targets.dino),
            var"local"=objective.dino_local(outputs.dino.local, data.targets.dino)
        ),
        ibot=objective.ibot(outputs.ibot, WHCVB_to_CWxHVB(data.inputs.mask), data.targets.ibot),
        koleo=(
            objective.koleo(@view(outputs.koleo[:, 1, :])) +
            objective.koleo(@view(outputs.koleo[:, 2, :]))
        )
    )

    loss =
        states.objective.weights.dino.global * scale_global * losses.dino.global +
        states.objective.weights.dino.local * scale_local * losses.dino.local +
        states.objective.weights.koleo * losses.koleo +
        states.objective.weights.ibot * losses.ibot

    states = (; student=states_student, objective=states.objective)
    # statistics = losses
    return loss, states, losses
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
