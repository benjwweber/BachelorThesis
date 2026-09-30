struct iBOTLoss
    temperature
end

iBOTLoss(; temperature) = iBOTLoss(temperature)

function (ibotloss::iBOTLoss)(
    patches_student_global,
    masks_student_global,
    patches_teacher_global,
)
    return apply_ibotloss(ibotloss, patches_student_global, masks_student_global, patches_teacher_global)
end

function apply_ibotloss(
    ibotloss,
    patches_student_global, # [Color, Width & Height, Crops & Batch]
    masks_student_global, # [1, Width & Height, Crops & Batch]
    patches_teacher_global # [Color, Width & Height, Crops & Batch]
)
    # masks_student_global = flatten(masks_student_global) # [1, Width & Height, Crops & Batch]
    temperature = convert(Lux.unwrapped_eltype(patches_student_global), ibotloss.temperature)
    loss = sum(
        patches_teacher_global .* Lux.NNlib.logsoftmax(
            patches_student_global ./ temperature;
            dims=1
        );
        dims=1
    ) # [1, Width × Height, Crops & Batch]
    loss = LuxLib.Impl.dropout_dot_mul(loss, masks_student_global) # [1, Width × Height, Crops & Batch]
    # loss = loss .* masks_student_global
    loss = sum(loss; dims=2) ./ max.(sum(masks_student_global; dims=2), 1) # [1, 1, Crops & Batch]
    loss = mean(-, loss) # fused_agg(mean, -, loss)

    return loss
end
