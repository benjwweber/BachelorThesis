using DataAugmentation
using CoordinateTransformations
using Distributions

struct RandomSizedCrop{
    N,
    S<:Distributions.Sampleable,
    R<:Distributions.Sampleable
} <: DataAugmentation.AbstractCrop
    dist_scale::S
    dist_ratio::R
end

RandomSizedCrop(; scale, ratio=(3 / 4, 4 / 3)) = RandomSizedCrop{2}(; scale, ratio)
function RandomSizedCrop{N}(; scale, ratio=(3 / 4, 4 / 3)) where {N}
    s = Uniform(scale...)
    r = LogUniform(ratio[1], ratio[2])
    return RandomSizedCrop{N,typeof(s),typeof(r)}(s, r)
end

function DataAugmentation.getrandstate(crop::RandomSizedCrop{N}) where {N}
    scale = rand(crop.dist_scale)
    ratio = rand(crop.dist_ratio)
    offsets = Tuple(rand() for _ in 1:N)
    return (scale, ratio, offsets)
end

function DataAugmentation.apply(crop::RandomSizedCrop, item::DataAugmentation.Item; randstate=DataAugmentation.getrandstate(crop))
    return DataAugmentation.apply(
        DataAugmentation.Project(CoordinateTransformations.IdentityTransformation()) |> crop,
        item;
        randstate=(nothing, (randstate,)))
end

function DataAugmentation.cropbounds(
    crop::RandomSizedCrop,
    bounds::Bounds;
    randstate=DataAugmentation.getrandstate(crop)
)
    scale, ratio, offsets = randstate

    h, w = length.(bounds.rs)
    a = h * w
    area = a * scale
    height = min(Int(round(sqrt(area * ratio))), h)
    width = min(Int(round(sqrt(area / ratio))), w)
    size = (height, width)
    bounds = DataAugmentation.offsetcropbounds(size, bounds, offsets)
    return bounds
end
