# should do @wrapperlayer and convert this to Chain?
@containerlayer struct AbMIL
	dense_in
	dropout_in
	attention
	dense_out
	dropout_out
end

function AbMIL(
	(dims_in, (dims_2, (dims_3, dims_out)));
	probability_dropout_in = 0.0f0,
	probability_dropout_out = 0.0f0,
	kwargs_dense_in = (;),
	kwargs_attention = (;),
	kwargs_dense_out = (;),
)
	return AbMIL(
		Dense(dims_in => dims_2, relu; kwargs_dense_in...),
		Dropout(probability_dropout_in),
		Layers.GatedAttention(dims_2 => dims_3; kwargs_attention...),
		Dense(dims_2 => dims_out; kwargs_dense_out...),
		Dropout(probability_dropout_out),
	)
end

@apply function (abmil::AbMIL)(
	inputs, # |> size == (nfeatures, ninstances, nbatches)
	masks # |> size == (1, ninstances, nbatches)
)
	return (
			inputs |> abmil.dense_in |> abmil.dropout_in,
			masks
		) |> abmil.attention |> abmil.dense_out |> abmil.dropout_out
end
