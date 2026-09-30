@containerlayer struct SkipConnection
    layers
    $connection
end

@apply function (layer::SkipConnection)(x)
    return layer.connection(layer.layers(x), x)
end

@apply function (layer::SkipConnection)(x, sines, cosines)
    return layer.connection(layer.layers(x, sines, cosines), x)
end

@containerlayer struct TransformerBlock
    skipconnection_attention
    skipconnection_feedforward
end

@apply function (block::TransformerBlock)(inputs, embeddings)
    outputs = (inputs, embeddings) |>
        block.skipconnection_attention |>
        block.skipconnection_feedforward
    return outputs, embeddings
end

function TransformerBlock(
    dim::Int,
    activation=Lux.NNlib.gelu;
    nheads::Int,
    ratio=4,
    probability_dropout_path=0.0f0,

    layer_norm=Lux.LayerNorm,
    kwargs_norm=(;),

    layer_attention=MultiHeadSelfAttention,
    kwargs_attention=(;),

    layer_feedforward=MultiLayerPerceptron,
    kwargs_feedforward=(;),

    # layer_scale=Lux.Scale,
    # kwargs_scale=(;
    #     use_bias=false,
    #     init_weight=constants32(1f-5)
    # ),

    layer_scale=Lux.Scale, #LayerScale,
    kwargs_scale=(;),
    kwargs_skipconnection=(;)
)
    norm = layer_norm(
        (dim, 1);  # shape excluding batch dimension, size(x) == (C, W × H + ..., B)
        dims=1,
        kwargs_norm...
    )
    attention = layer_attention(
        dim;
        nheads=nheads,
        kwargs_attention...
    )
    feedforward = layer_feedforward(
        dim => Int(round(dim * ratio)) => dim,
        activation;
        kwargs_feedforward...
    )
    scale = layer_scale(
        (dim, 1);
        kwargs_scale...
    )
    # skipconnection_attention = PathDropoutSkipConnection(
    # 	AttentionBlock(
    # 		norm,
    # 		attention,
    # 		scale
    # 	),
    # 	+,
    # 	probability_dropout_path;
    # 	kwargs_skipconnection...
    # )
    # skipconnection_feedforward = PathDropoutSkipConnection(
    # 	FeedForwardBlock(
    # 		norm,
    # 		feedforward,
    # 		scale
    # 	),
    # 	+,
    # 	probability_dropout_path;
    # 	kwargs_skipconnection...
    # )
    skipconnection_attention = Layers.SkipConnection(
        Lux.Chain(;
            block=AttentionBlock(
                norm,
                attention,
                scale
            ),
            dropout=Lux.Dropout(probability_dropout_path; dims=4)
        ),
        +ᵢ;
        kwargs_skipconnection...
    )
    skipconnection_feedforward = Layers.SkipConnection(
        Lux.Chain(;
            block=FeedForwardBlock(
                norm,
                feedforward,
                scale
            ),
            dropout=Lux.Dropout(probability_dropout_path; dims=4)
        ),
        +;
        kwargs_skipconnection...
    )
    return TransformerBlock(
        skipconnection_attention,
        skipconnection_feedforward
    )
end

@containerlayer struct AttentionBlock
    norm
    attention
    scale
end

@apply function (block::AttentionBlock)(inputs, embeddings)
    return (inputs |> block.norm, embeddings) |> block.attention |> first |> block.scale
end

@containerlayer struct FeedForwardBlock
    norm
    feedforward
    scale
end

@apply function (block::FeedForwardBlock)(inputs)
    return inputs |> block.norm |> block.feedforward |> block.scale
end

function (+ᵢ)(inputs, (outputs, embeddings))
    return inputs + outputs
end
