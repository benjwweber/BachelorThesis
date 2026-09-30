struct DINOSet
    data::TileLoader
    transforms_views_global
    transforms_masks_global
    transforms_views_local
    config
end

function DINOSet(
    data::TileLoader;
    dims_global::Dims{2},
    scale_global,
    nviews_global=2,
    dims_local::Dims{2},
    scale_local,
    nviews_local::Int,
    dims_mask,
    probability_mask,
    npatches_min,
    npatches_max,
    area_min,
    area_max,
    ratio_min,
    ratio_max,
)
    transform_flip = RandomApply(FlipDim{2}(1), 0.5) |> RandomApply(FlipDim{2}(2), 0.5)
    transform_crop_global = RandomSizedCrop(; scale=scale_global) |> ScaleFixed(dims_global)
    transform_crop_local = RandomSizedCrop(; scale=scale_local) |> ScaleFixed(dims_local)
    transform_color =
        RandomApply(
            RandomSequence(
                AdjustBrightness(0.4),
                AdjustContrast(0.4),
                AdjustSaturation(0.2),
                AdjustHue(36)
            ),
            0.8
        ) |>
            RandomApply(Grayscale(), 0.2)

    transform_normalize =
        ColorToChannels() |>
            ToEltype(Float32) |>
            Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            ) |>
            ToEltype(Float32)

    transform_crop_global_1 =
        transform_flip |>
            transform_crop_global |>
            PinOrigin() |>
            transform_color |>
            GaussianBlur() |>
            transform_normalize

    transform_crop_global_2 =
        transform_flip |>
            transform_crop_global |>
            PinOrigin() |>
            transform_color |>
            RandomApply(GaussianBlur(), 0.1) |>
            RandomApply(Solarize(0.5), 0.2) |>
            transform_normalize

    transforms_views_global = [transform_crop_global_1, transform_crop_global_2]

    transforms_views_local = [
        transform_flip |>
            transform_crop_local |>
            PinOrigin() |>
            transform_color |>
            RandomApply(GaussianBlur(), 0.5) |>
            transform_normalize
        for _ in 1:nviews_local
    ]

    transforms_masks_global = [
        OneOf(
            [
                MaskGenerator(
                    dims_mask;
                    npatches_min,
                    npatches_max,
                    area_min,
                    area_max,
                    ratio_min,
                    ratio_max
                ),
                NoMaskGenerator(dims_mask)
            ],
            [probability_mask, 1 - probability_mask]
        )
        for _ in transforms_views_global
    ]

    config = (;
        dims_global,
        scale_global,
        nviews_global,
        dims_local,
        scale_local,
        nviews_local,
        dims_mask,
        probability_mask,
        npatches_min,
        npatches_max,
        area_min,
        area_max,
        ratio_min,
        ratio_max,
    )

    return DINOSet(data, transforms_views_global, transforms_masks_global, transforms_views_local, config)
end

Base.length(dinoset::DINOSet) = length(dinoset.data)
numobs(dinoset::DINOSet) = numobs(dinoset.data)

function Base.getindex(dinoset::DINOSet, index::Int)
    item = Image(dinoset.data[index])

    views_global = MLUtils.batch([
        itemdata(DataAugmentation.apply(transform, item))
        for transform in dinoset.transforms_views_global
    ])

    mask = MLUtils.batch([
        itemdata(DataAugmentation.apply(transform, item))
        for transform in dinoset.transforms_masks_global
    ])

    views_local = MLUtils.batch([
        itemdata(DataAugmentation.apply(transform, item))
        for transform in dinoset.transforms_views_local
    ])

    return (; views=(; var"global"=views_global, var"local"=views_local), mask)
end


function Serialization.serialize(s::AbstractSerializer, d::DINOSet)
    Serialization.serialize_type(s, DINOSet)
    Serialization.serialize(s, d.data)
    Serialization.serialize(s, d.config)
    return nothing
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{DINOSet})
    data = Serialization.deserialize(s)
    config = Serialization.deserialize(s)
    return DINOSet(
        data;
        config...
    )
end