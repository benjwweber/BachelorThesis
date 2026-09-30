struct Normalize2 <: Lux.AbstractLuxLayer
    dims
    epsilon
end

Normalize2(dims; epsilon) = Normalize2(dims, epsilon)

function (layer::Normalize2)(x, parameters, states)
    y = normalize2(x; dims=layer.dims, epsilon=convert(Lux.unwrapped_eltype(x), layer.epsilon))
    return y, states
end

function Base.show(io::IO, n::Normalize2)
    print(io, "Normalize2(")
    n.dims != Colon() && print(io, "dims=", n.dims)
    return print(io, ")")
end
