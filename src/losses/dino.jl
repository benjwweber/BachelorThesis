@concrete struct DINOLoss
    temperature
    ignore_diagonal
end
DINOLoss(; temperature, ignore_diagonal=Val(true)) = DINOLoss(temperature, ignore_diagonal)

function (loss::DINOLoss)(
    logits_student, # [Color, Crop, Batch]
    probabilities_teacher, # [Color, Crop, Batch]
)
    return compute_dinoloss(
        loss,
        logits_student,
        probabilities_teacher,
        loss.ignore_diagonal
    )
end

function compute_dinoloss(
    dinoloss,
    logits_student, # [Color, Crop, Batch]
    probabilities_teacher, # [Color, Crop, Batch]
    ignore_diagonal::Val{false}
)
    batchsize = size(logits_student, 3)
    ncrops_student = size(logits_student, 2)
    ncrops_teacher = size(probabilities_teacher, 2)
    temperature = convert(Lux.unwrapped_eltype(logits_student), dinoloss.temperature)

    probabilities_student = Lux.NNlib.logsoftmax(
        logits_student ./ temperature;
        dims=1
    )
    loss = sum(
        -,
        (
            insertdims(probabilities_teacher; dims=2) # [Color, 1, Crop, Batch]
            .*
            insertdims(probabilities_student; dims=3) # [Color, Crop, 1, Batch]
        ) # [Color, Crop_student, Crop_teacher, Batch]
    )
    return loss / (batchsize * ncrops_student * ncrops_teacher)
end


function compute_dinoloss(
    dinoloss,
    logits_student, # [Color, Crop, Batch]
    probabilities_teacher, # [Color, Crop, Batch]
    ignore_diagonal::Val{true}
)
    batchsize = size(logits_student, 3)
    ncrops_student = size(logits_student, 2)
    ncrops_teacher = size(probabilities_teacher, 2)
    temperature = convert(Lux.unwrapped_eltype(logits_student), dinoloss.temperature)

    probabilities_student = Lux.NNlib.logsoftmax(
        logits_student ./ temperature;
        dims=1
    )
    loss = dropdims(
        sum(
            (
                insertdims(probabilities_teacher; dims=2) # [Color, 1, Crop, Batch]
                .*
                insertdims(probabilities_student; dims=3) # [Color, Crop, 1, Batch]
            ); # [Color, Crop_student, Crop_teacher, Batch]
            dims=(1, 4)
        ); # [1, Crop_student, Crop_teacher, 1]
        dims=(1, 4) # [Crop_student, Crop_teacher]
    )
    ncrops_min = min(ncrops_student, ncrops_teacher)
    # @views loss[diagind(loss)] .= 0
    # return sum(-, loss) / (batchsize * ncrops_student * ncrops_teacher - batchsize * ncrops_min)
    return (sum(-, loss) + sum(loss[diagind(loss)])) / (batchsize * ncrops_student * ncrops_teacher - batchsize * ncrops_min)
end
