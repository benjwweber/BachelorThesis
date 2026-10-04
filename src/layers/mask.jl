# embedding instead of token in states !!!

@concrete struct MaskToken <: Lux.AbstractLuxLayer
	dims
	init
end

MaskToken(dims; init = Lux.zeros32) = MaskToken(dims, init)

Lux.LuxCore.initialparameters(rng::AbstractRNG, layer::MaskToken) = (;
	token = layer.init(rng, layer.dims..., 1) # embedding instead of token in states !!!
)
Lux.LuxCore.parameterlength(layer::MaskToken) = prod(layer.dims)
Lux.LuxCore.initialstates(::AbstractRNG, ::MaskToken) = (;training = Val(true))
# Lux.LuxCore.statelength(::MaskToken) = 1


function (layer::MaskToken)((inputs, masks), parameters, states)
	outputs = apply_mask(inputs, masks, parameters.token, states.training)
	return outputs, states
end

function apply_mask(inputs, masks, token, training::Val{true})
	# ! might need to change masker for this !
	return ifelse.(masks, inputs, token)
	# return LuxLib.Impl.dropout_dot_mul(inputs, .!masks) .+ LuxLib.Impl.dropout_dot_mul(token, masks)
end

function apply_mask(inputs, masks::Nothing, token, training::Val{true})
	return inputs
end

function apply_mask(inputs, masks, token, training::Val{false})
	return inputs
end

function Base.show(io::IO, m::MaskToken)
    return print(io, "MaskToken(", m.dims, ")")
end
