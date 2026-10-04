using MLStyle

function construct_layers(layers, expr)
	MLStyle.@match expr begin
		Expr(:$, field) => field
		field::Symbol => begin push!(layers, QuoteNode(field)); field end
		::Any => expr
	end
end

macro containerlayer(expr)
	layers = []
	layer = expr.args[2]
	fields = construct_layers.(Ref(layers), expr.args[3].args)
	return esc(Expr(:struct, false, Expr(:(<:), layer, Expr(:curly, :(Lux.AbstractLuxContainerLayer), Expr(:tuple, layers...))), Expr(:block, fields...)))
end
