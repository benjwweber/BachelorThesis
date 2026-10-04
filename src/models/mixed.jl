struct MixedPrecision{P,O} <: Lux.AbstractLuxWrapperLayer{:layer}
    layer
end

function (layer::MixedPrecision{P,O})(input, parameters, states) where {P,O}
    output_p, states_new = Lux.apply(
        layer.layer,
        input, #fmap(to(P), input),
        fmap(to(P), parameters),
        fmap(to(P), states)
    )
    output = fmap(to(O), output_p)
    return output, states_new
end

to(type) = Base.Fix1(to, type)
to(type, x::AbstractArray{<:Number}) = type.(x) # convert(AbstractArray{type}, x)
# Lux.Utils.ofeltype_array(type, x) # aaaahh Number important because Traced...
to(type, x) = x
