
tobf16(x) = x
tobf16(x::AbstractArray{<:Number}) = BFloat16.(x)
tof32(x) = x
tof32(x::AbstractArray{<:Number}) = Float32.(x)

function constants32(args...; value, kwargs...)
    x = Lux.ones32(args...; kwargs...)
    x .= value
    return x
end

function constants32(value)
    return Lux.WeightInitializers.PartialFunction.Partial{Missing}(constants32, nothing, pairs((; value)))
end

function flatten(
    inputs # |> size == (W, H, C, B)
)
    return permutedims(
        reshape(inputs, :, size(inputs, 3), size(inputs, 4)), # |> size == (W×H, D, B)
        (2, 1, 3)
    ) # |> size == (D, W×H, B)
end

# #  julia +1.12
# function constants32(value)
# 	return Base.Fix1(constants32, value)
# end

# function constants32(value, args...; kwargs...)
# 	x = Lux.ones32(args...; kwargs...)
# 	x .= value
# 	return x
# end

function norm2(x; dims)
    return sqrt.(sum(abs2, x; dims))
end

function normalize2(x; dims=1, epsilon=1f-8)
    return @. x / max($norm2(x; dims), epsilon)
end

function flatten_dims_batch(inputs; ndims=1)
    sizes = size(inputs)
    return reshape(inputs, sizes[1:(end-ndims)]..., :), sizes[(end-ndims+1):end]
end


function reconstruct_dims_batch(inputs; dims)
    return reshape(inputs, size(inputs)[1:end-1]..., dims...)
end


if VERSION < v"1.12"
    function insertdims(inputs; dims::Int)
        dims_ = size(inputs)
        return reshape(inputs, dims_[begin:dims-1]..., 1, dims_[dims:end]...)
    end
end

# # julia +1.11
# function constants32(value)
# 	return Lux.ones32
# end

# # FOR EVERYTHING TO WORK IN julia +1.11
# function insertdims()
# 	return
# end

batchdim(x) = ndims(x)
#batchsize(x) = size(x, batchdim(x))

function reversedims(x)
    return permutedims(x, reverse(ntuple(identity, ndims(x))))
end

function imagestotensor(images)
    tensors = images |> Map(reversedims ∘ channelview) |> tcollect
    tensor = MLUtils.batch(tensors)
    return tensor
end

Base.getproperty(property::Symbol) = Base.Fix2(getproperty, property)

toval(value) = Val(value)
toval(value::Val) = value

macro comment_str(str)
    return LineNumberNode(0, str)
end

flatten_tail(inputs) = reshape(inputs, size(inputs, 1), :)
function flatten_last(inputs, ndims=1)
    sizes = size(inputs)
    return reshape(inputs, sizes[1:(end-ndims)]..., :)#, sizes[(end - ndims + 1):end]
end

function flatten_last(::Nothing, ndims=1)
    sizes = size(inputs)
    return reshape(inputs, sizes[1:(end-ndims)]..., :)#, sizes[(end - ndims + 1):end]
end

function reshape_last(inputs, dims...)
    sizes = size(inputs)
    return reshape(inputs, sizes[1:(end-1)]..., dims...)
end
reshape_last(inputs, (dims...,)) = reshape_last(inputs, dims...)
