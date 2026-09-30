@containerlayer struct MultiLayerPerceptron
	dense_in
	dropout_in
	dense_out
	dropout_out
end

@apply function (mlp::MultiLayerPerceptron)(inputs)
	return inputs |> mlp.dense_in |> mlp.dropout_in |> mlp.dense_out |> mlp.dropout_out
end

function MultiLayerPerceptron(
	dims,
	activation = Lux.NNlib.gelu, # Lux.NNlib.gelu_erf,
	probability_dropout = 0.0f0;
	kwargs_dense = (; use_bias = true),
	kwargs_dropout = (; )
)
	dims = parse_mlp_dims(dims)
	MultiLayerPerceptron(
		Lux.Dense(dims.in => dims.hidden, activation; kwargs_dense...),
		Lux.Dropout(probability_dropout; kwargs_dropout...),
		Lux.Dense(dims.hidden => dims.out; kwargs_dense...),
		Lux.Dropout(probability_dropout; kwargs_dropout...)
	)
end

parse_mlp_dims(in) = parse_mlp_dims((in, (in, in)))
parse_mlp_dims((in, out)) = parse_mlp_dims((in, (in, out)))
parse_mlp_dims((in, (hidden, out))) = (; in = in, hidden = hidden, out = out)

@containerlayer struct GatedLinearUnit
	dense_in_1
	dense_in_2
	dense_out
end

function GatedLinearUnit(
	dims,
	activation = Lux.swish,
	probability_dropout = 0.0f0;
	kwargs_dense = (; use_bias = true),
	align::Int = 8
)
	dims = parse_dims(dims; align = align)
	return GatedLinearUnit(
		Lux.Dense(dims.in => dims.hidden, activation; kwargs_dense...),
		Lux.Dense(dims.in => dims.hidden; kwargs_dense...),
		Lux.Dense(dims.hidden => dims.out; kwargs_dense...)
	)
end

@apply function (glu::GatedLinearUnit)(inputs)
	return glu.dense_out(glu.dense_in_1(inputs) .* glu.dense_in_2(inputs))
end

parse_glu_dims(in; align) = parse_glu_dims((in, (in , in)); align = align)
parse_glu_dims((in, out); align) = parse_glu_dims((in, (in , out)); align = align)
function parse_glu_dims((in, (hidden , out)); align)
	d = Int(round(hidden * 2//3))
	hidden = d + (-d % align)
	return (; in = in, hidden = hidden, out = out)
end
