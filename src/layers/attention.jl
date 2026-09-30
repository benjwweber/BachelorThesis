@containerlayer struct MultiHeadSelfAttention
	$nheads
	dense_q
	dense_k
	dense_v
	dropout_attention
	dense_out
	dropout_out
end

@apply function (mhsa::MultiHeadSelfAttention)(
	inputs, # |> size == (Width, Height, Color, Batch) || (Width, Height, Color, Crop, Batch)
	embeddings
)
	q = mhsa.dense_q(inputs)
	    # |> size == (Color, Width × Height + Prefix, ...)
	k = mhsa.dense_k(inputs)
	v = mhsa.dense_v(inputs)

	q = reshape(q, size(q, 1) ÷ mhsa.nheads, mhsa.nheads, size(q)[2:end]...)
	    # |> size == (Color ÷ nheads, Head, Width × Height + Prefix, ...)
	k = reshape(k, size(k, 1) ÷ mhsa.nheads, mhsa.nheads, size(k)[2:end]...)
	v = reshape(v, size(v, 1) ÷ mhsa.nheads, mhsa.nheads, size(v)[2:end]...)

	q = apply_rotarypositionalembedding(q, embeddings)
	    # |> size == (Color ÷ nheads, Head, Width × Height + Prefix, ...)
	k = apply_rotarypositionalembedding(k, embeddings)

	dropout_attention_stateful = Lux.StatefulLuxLayer(
		mhsa.dropout_attention, parameters.dropout_attention, states.dropout_attention
	)

	outputs_attention, α = Lux.LuxLib.API.scaled_dot_product_attention(q, k, v; fdrop = dropout_attention_stateful) #= , head_dim = 1, token_dim = 3) =#

	states_dropout_attention = dropout_attention_stateful.st
	outputs_attention = reshape(
		outputs_attention,
		size(outputs_attention, 1) * mhsa.nheads, size(outputs_attention)[3:end]...
	) # |> size == (Color, Width × Height + Prefix, ...)

	outputs = outputs_attention |> mhsa.dense_out |> mhsa.dropout_out
	return outputs, embeddings
end

function parse_mhsa_dims(dims)
	return (; in_q = dims, in_k = dims, in_v = dims, qk = dims, v = dims, out = dims)
end

function MultiHeadSelfAttention(
	dims;
	nheads,
	kwargs_dense_qkv = (; use_bias = true),
	kwargs_dense_q = (;),
	kwargs_dense_k = (;), # mask_k_bias = false => (; use_bias = false)
	kwargs_dense_v = (;),
	kwargs_dense_out = (; use_bias = true),
	probability_dropout_attention = 0.0f0,
	probability_dropout_out = 0.0f0,
	# is_causal::Union{Bool,Nothing} = nothing,
)
	dims = parse_mhsa_dims(dims)
	@argcheck dims.qk % nheads == 0
	@argcheck dims.v % nheads == 0

	return MultiHeadSelfAttention(
		nheads,
		Lux.Dense(dims.in_q, dims.qk; merge(kwargs_dense_qkv, kwargs_dense_q)...),
		Lux.Dense(dims.in_k, dims.qk; merge(kwargs_dense_qkv, kwargs_dense_k)...),
		Lux.Dense(dims.in_v, dims.v; merge(kwargs_dense_qkv, kwargs_dense_v)...),
		Lux.Dropout(probability_dropout_attention),
		Lux.Dense(dims.v, dims.out; kwargs_dense_out...),
		Lux.Dropout(probability_dropout_out),
	)
end

@containerlayer struct GatedAttention
	dense_V
	dropout_V
	dense_U
	dropout_U
	dense_w
end

function GatedAttention(
	(dims_in, dims_attention);
	probability_dropout_U = 0.0f0,
	probability_dropout_V = 0.0f0,
	kwargs_dense_V = (;),
	kwargs_dense_U = (;),
	kwargs_dense_w = (;),
)
	return GatedAttention(
		Dense(dims_in => dims_attention, tanh; kwargs_dense_V...),
		Dropout(probability_dropout_V),

		Dense(dims_in => dims_attention, sigmoid; kwargs_dense_U...),
		Dropout(probability_dropout_U; kwargs_dense_w...),

		Dense(dims_attention => 1),
	)
end

@apply function (layer::GatedAttention)(
	inputs, # |> size == (Color, Bag, Batch)
	masks # |> size == (1, Bag, Batch)
)
	# if i dont output α, this can be moved into the matmul
	z = layer.dense_w(layer.dropout_V(layer.dense_V(inputs)) .* layer.dropout_U(layer.dense_U(inputs))) # |> size == (1, Bag, Batch)
	a = softmax(
		ifelse.(masks, z, typemin(eltype(z)));
		dims = 2
	)
	# a = softmax(a .- (.!masks .* prevfloat(typemax(eltype(inputs)))); dims = 2)
	outputs = batched_matmul(
		inputs, a;
		rhs_contracting_dim = 2, # this is the same as inputs * transpose(α)
		lhs_contracting_dim = 2
	) # |> size == (Color, 1, Batch)
	return outputs
end