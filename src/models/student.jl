@containerlayer struct Student
    backbone
    head_dino
    head_ibot
end

@apply function (student::Student)(
    inputs #(; views = (; global, local), mask)
)

    nprefix = student.backbone.token_prefix.dims[2]
    nbatches = size(inputs.views.global, 5)
    nviews_global = size(inputs.views.global, 4)
    nviews_local = size(inputs.views.local, 4)

    output_global = student.backbone(
        flatten_last(inputs.views.global, 2),
        flatten_last(inputs.mask, 2)
    )
    output_local = student.backbone(
        flatten_last(inputs.views.local, 2),
        nothing
    )

    class_global = @view output_global[:, 1, :]
    class_local = @view output_local[:, 1, :]
    patches_global = @view output_global[:, (nprefix+1):end, :]

    dino_global = reshape_last(
        student.head_dino(class_global), nviews_global, nbatches
    )
    dino_local = reshape_last(
        student.head_dino(class_local), nviews_local, nbatches
    )
    return (;
        dino=(
            var"global"=dino_global,
            var"local"=dino_local
        ),
        ibot=reshape_last(
            student.head_ibot(patches_global), nviews_global, nbatches
        ),
        koleo=reshape_last(class_global, nviews_global, nbatches)
    )
end
