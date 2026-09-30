@containerlayer struct RotaryPositionalEmbedding
    $dim
    $nheads
    $init_periods
    $normalize
    shift # never used?
    jitter # never used?
    rescale # used?
end

@apply periods function (layer::RotaryPositionalEmbedding)(
    inputs # |> size == (Width, Height, Color, Batch) || (Width, Height, Color, Crop, Batch)
)
    width = size(inputs, 1)
    height = size(inputs, 2)

    coords_height = similar(inputs, height)
    @. coords_height = 2 * ($range(0.5f0, height) / $(layer.normalize)(height, width)) - 1

    coords_width = similar(inputs, width)
    @. coords_width = 2 * ($range(0.5f0, width) / $(layer.normalize)(width, height)) - 1

    coords = cat(
        repeat(coords_height, inner=width),
        repeat(coords_width, outer=height);
        dims=3
    ) # |> size == (Width × Height, 1, 2)

    coords = coords |> layer.shift |> layer.jitter |> layer.rescale
    ET = Lux.unwrapped_eltype(inputs)
    piET = convert(ET, pi)
    periodsET = Lux.Utils.ofeltype_array(ET, periods)
    angles = reversedims(
        reshape(
            (@. 2 * piET * coords / periodsET), # |> size == (Width × Height, Color ÷ 4, 2)
            (height * width, 1, :)
        ), # |> size == (Width × Height, 1, Color ÷ 2)
    ) # |> size == (Color ÷ 2, 1, Width × Height)

    cosines = cos.(angles) # |> size == (Color ÷ 2, 1, Width × Height)
    sines = sin.(angles) # 〃

    return (; sines, cosines)
end

function apply_rotarypositionalembedding(
    x,
    (; sines, cosines)
)
    nprefixtoken = size(x, 3) - size(sines, 3)
    dim_color_half = size(x, 1) ÷ 2
    dims_batch = ntuple(Returns(:), ndims(x) - 3)
    x_prefix = @view x[:, :, 1:nprefixtoken, dims_batch...]
    x_1 = @view x[1:dim_color_half, :, (nprefixtoken+1):end, dims_batch...]
    x_2 = @view x[(dim_color_half+1):end, :, (nprefixtoken+1):end, dims_batch...]

    x_left = @. x_1 * cosines - x_2 * sines
    x_right = @. x_2 * cosines + x_1 * sines
    y = cat(x_prefix, vcat(x_left, x_right); dims=3)
    return y
end

function apply_rotarypositionalembedding(x, ::Tuple{Nothing,Nothing})
    return x
end

Lux.Experimental.layer_map_leaf(::KeyPath, ::RotaryPositionalEmbedding) = true

function RotaryPositionalEmbedding(
    dim;
    nheads,
    args_init_periods=(; base=100), # else (;min = , max = )
    normalize=first,
    factor_shift=nothing,
    factor_jitter=nothing,
    factor_rescale=nothing,
)
    #@assert normalization in (:min, :max, :separate)
    return RotaryPositionalEmbedding(
        dim,
        nheads,
        Base.Fix1(init_periods, args_init_periods),
        normalize, # Base.Fix1(normalize, Val(normalization)), # normalize(Val(normalization)), # +1.11
        PLACEHOLDER((1, 1, 2), factor_shift, identity),
        PLACEHOLDER((1, 1, 2), factor_jitter, log),
        PLACEHOLDER((1, 1, 1), factor_rescale, log)
    )
end

function Lux.initialstates(rng::AbstractRNG, layer::RotaryPositionalEmbedding)
    return (;
        shift=Lux.initialstates(rng, layer.shift),
        jitter=Lux.initialstates(rng, layer.jitter),
        rescale=Lux.initialstates(rng, layer.rescale),
        periods=layer.init_periods(layer),
    )
end

function Lux.LuxCore.statelength(layer::RotaryPositionalEmbedding)
    return (layer.dim ÷ layer.nheads) ÷ 4 +
           Lux.LuxCore.statelength(layer.shift) +
           Lux.LuxCore.statelength(layer.jitter) +
           Lux.LuxCore.statelength(layer.rescale)
end

function init_periods((; base)::@NamedTuple{base::Int}, layer::RotaryPositionalEmbedding)
    dim_per_head = (layer.dim ÷ layer.nheads)
    periods = base .^ (2 * range(0.0f0, dim_per_head ÷ 4 - 1) / (dim_per_head ÷ 2))
    periods = reshape(periods, 1, size(periods)...)#insertdims(periods, dims = 1)
    return periods # |> size == (1, D ÷ 4)
end

function init_periods((; min, max)::@NamedTuple{min::Int, max::Int}, layer::RotaryPositionalEmbedding)
    dim_per_head = (layer.dim ÷ layer.nheads)
    base = max / min
    exponents = range(0.0f0, 1.0f0; length=dim_per_head ÷ 4)
    periods = ((base .^ exponents) / base) * layer.max
    periods = reshape(periods, 1, size(periods)...) # +1.12 insertdims(periods, dims = 1)
    return periods # |> size == (1, D ÷ 4)
end

@concrete struct PLACEHOLDER <: Lux.AbstractLuxLayer
    dims
    factor
    f
    g
end

PLACEHOLDER(dims, factor, f=identity) = PLACEHOLDER(dims, factor, f, inverse(f))
PLACEHOLDER(dims, ::Nothing, f, g) = Lux.NoOpLayer()

Lux.initialparameters(::AbstractRNG, ::PLACEHOLDER) = NamedTuple()
Lux.LuxCore.parameterlength(::PLACEHOLDER) = 0

Lux.initialstates(rng::AbstractRNG, layer::PLACEHOLDER) = (;
    rng=Lux.Utils.sample_replicate(rng),
    training=Val(true)
)
Lux.LuxCore.statelength(::PLACEHOLDER) = 2

@apply (rng::AbstractRNG, training::Val{true}) function (layer::PLACEHOLDER)(inputs)
    factor = layer.f(layer.factor)
    rands = similar(inputs, layer.dims)
    rand!(rng, rands)
    outputs = @. inputs * layer.g(2 * factor * rands - factor)
    return outputs
end

@apply (rng, training::Val{false}) function (layer::PLACEHOLDER)(x)
    return x
end
