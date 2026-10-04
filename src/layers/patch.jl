@containerlayer struct PatchEmbedding
	conv
	norm
end

function PatchEmbedding(
	dims_patch::Dims,
	(dim_channel, dim_embedding);
	kwargs_conv = (;
		cross_correlation = true # for PyTorchs compatibility
	),
	layer_norm = Returns(Lux.NoOpLayer()),
	kwargs_norm = (; )
)
	return PatchEmbedding(
		Lux.Conv(
			dims_patch,
			dim_channel => dim_embedding;
			stride = dims_patch,
			kwargs_conv...
		),
		layer_norm((1, 1, dims_patch); dims = 3, kwargs_norm...)
	)
end

@apply function (layer::PatchEmbedding)(inputs)
	return inputs |> layer.conv |> layer.norm
end
