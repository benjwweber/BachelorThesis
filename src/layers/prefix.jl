# embedding instead of token in states !!!

using ChainRulesCore: @non_differentiable

@concrete struct PrefixToken <: Lux.AbstractLuxLayer
	dims
	init
end

Lux.Experimental.layer_map_leaf(::KeyPath, ::PrefixToken) = true

function PrefixToken(
	dim_embedding::Int,
	ntoken_register::Int,
	ntoken_class::Int = 1;
	init = zeros32
)
	return PrefixToken(
		(dim_embedding, ntoken_register + ntoken_class),
		init
	)
end

Lux.LuxCore.initialparameters(rng::AbstractRNG, prefix::PrefixToken) = (;
	token = prefix.init(rng, prefix.dims..., 1)
)
Lux.LuxCore.parameterlength(prefix::PrefixToken) = prod(prefix.dims)

function onesbatch(x::AbstractArray{T, N}, dims) where {T, N}
	return fill!(similar(x, dims..., size(x)[3:end]...), one(T))
end
@non_differentiable onesbatch(x::AbstractArray, dims...)

function prefixbatch(x, token)
	ndims_token = ndims(token)
	return repeat(token, ntuple(Returns(1), ndims_token - 1)..., size(x, ndims_token))
	#return token .* onesbatch(x, size(token))
end

function (prefix::PrefixToken)(inputs, parameters, states)
	outputs = cat(prefixbatch(inputs, parameters.token), inputs; dims = Val(2))
	return outputs, states
end

function Base.show(io::IO, p::PrefixToken)
    return print(io, "PrefixToken(", p.dims, ")")
end
