@containerlayer struct VisionTransformer
    embedding_patch
    token_mask
    embedding_rotarypositional
    token_prefix
    blocks
    norm
    # head
end

"""
test
"""
@apply function (vit::VisionTransformer)(
    inputs
)
    patches = inputs |> vit.embedding_patch
    embeddings = patches |> vit.embedding_rotarypositional
    token = patches |> WHCB_to_CWxHB |> vit.token_prefix
    outputs = (token, embeddings) |> vit.blocks |> first |> vit.norm
    return outputs
end

@apply function (vit::VisionTransformer)(
    inputs, # |> size == (W⋅k, H⋅k, 3, B) && k == dims_patch
    masks # |> size == (W, H, 1, B)
)
    # (W⋅k, H⋅k, 3, B) => (W, H, E, B)
    patches = inputs |> vit.embedding_patch

    # (W, H, E, B) => (E ÷ 2, 1, W × H, 1), (E ÷ 2, 1, W × H, 1)
    embeddings = patches |> vit.embedding_rotarypositional

    # (W, H, E, B), (W, H, 1, B) => (W, H, E, B) => (E, W×H, B) => (E, W×H + P, B)
    token = (patches, masks) |> vit.token_mask |> WHCB_to_CWxHB |> vit.token_prefix

    # (E, W×H + P, B), ... => (E, W×H + P, B), ... => (E, W×H + P, B) => (E, W×H + P, B)
    outputs = (token, embeddings) |> vit.blocks |> first |> vit.norm
    return outputs
end

@apply function (vit::VisionTransformer)(
    inputs, # |> size == (W⋅k, H⋅k, 3, B) && k == dims_patch
)
    # (W⋅k, H⋅k, 3, B) => (W, H, E, B)
    patches = inputs |> vit.embedding_patch

    # (W, H, E, B) => (E ÷ 2, 1, W × H, 1), (E ÷ 2, 1, W × H, 1)
    embeddings = patches |> vit.embedding_rotarypositional

    # (W, H, E, B), (W, H, 1, B) => (W, H, E, B) => (E, W×H, B) => (E, W×H + P, B)
    token = patches |> WHCB_to_CWxHB |> vit.token_prefix

    # (E, W×H + P, B), ... => (E, W×H + P, B), ... => (E, W×H + P, B) => (E, W×H + P, B)
    outputs = (token, embeddings) |> vit.blocks |> first |> vit.norm
    return outputs
end

function VisionTransformer(
    dims_patch,
    (dim_channel, dim_embedding);
    depth,
    nheads,
    ratio,
    ntoken_register=4,
    ntoken_class=1, kwargs_embedding_patch=(;),
    kwargs_embedding_rotarypositional=(;),
    kwargs_token_mask=(;),
    kwargs_token_prefix=(;),
    kwargs_blocks=(;),
    layer_norm=Lux.LayerNorm,
    kwargs_norm=(;)
)
    embedding_patch = Layers.PatchEmbedding(
        dims_patch,
        (dim_channel, dim_embedding);
        kwargs_embedding_patch...
    )
    token_mask = Layers.MaskToken(
        (1, 1, dim_embedding);
        kwargs_token_mask...
    )
    embedding_rotarypositional = Layers.RotaryPositionalEmbedding(
        dim_embedding;
        nheads=nheads,
        kwargs_embedding_rotarypositional...
    )
    token_prefix = Layers.PrefixToken(
        dim_embedding,
        ntoken_register,
        ntoken_class;
        kwargs_token_prefix...
    )
    blocks = Lux.Chain(;
        (Symbol("block_", i) => Layers.TransformerBlock(
            dim_embedding,
            Lux.NNlib.gelu;
            nheads=nheads,
            ratio=ratio,
            layer_norm=layer_norm,
            kwargs_blocks...
        ) for i in 1:depth)...
    )
    norm = layer_norm(
        (dim_embedding, 1);
        dims=1,
        kwargs_norm...
    )
    # head = Lux.NoOpLayer()
    return VisionTransformer(
        embedding_patch,
        token_mask,
        embedding_rotarypositional,
        token_prefix,
        blocks,
        norm,
        # head
    )
end
function WHCB_to_CWxHB(
    x # |> size == (W, H, C, B)
)
    return permutedims(
        reshape(
            x,
            (:, size(x, 3), size(x, 4))
        ), # |> size == (W×H, C, B)
        (2, 1, 3)
    ) # |> size == (C, W×H, B)
end

function vit_small(dims_patch=(16, 16); kwargs...)
    model = VisionTransformer(
        dims_patch=dims_patch,
        3 => 384,
        depth=12,
        nheads=6,
        ratio=4,
        kwargs...
    )
    return model
end

function vit_base(dims_patch=(16, 16); kwargs...)
    model = VisionTransformer(
        dims_patch,
        3 => 768;
        depth=12,
        nheads=12,
        ratio=4,
        kwargs...
    )
    return model
end

function vit_large(dims_patch=(16, 16); kwargs...)
    model = VisionTransformer(
        dims_patch,
        3 => 1024;
        depth=24,
        nheads=16,
        ratio=4,
        kwargs...
    )
    return model
end

function vit_so400m(dims_patch=(16, 16); kwargs...)
    model = VisionTransformer(
        dims_patch,
        3 => 1152;
        depth=27,
        nheads=18,
        ratio=34 // 9,
        kwargs...
    )
    return model
end

function vit_huge2(dims_patch=(16, 16); kwargs...)
    model = VisionTransformer(
        dims_patch,
        3 => 1280;
        depth=32,
        nheads=20,
        ratio=4,
        kwargs...
    )
    return model
end

function vit_giant2(dims_patch=(16, 16); kwargs...)
    model = VisionTransformer(
        dims_patch,
        3 => 1536;
        depth=40,
        nheads=24,
        ratio=4,
        kwargs...
    )
    return model
end

function vit_7b(dims_patch=(16, 16); kwargs...)
    model = VisionTransformer(
        dims_patch,
        3 => 4096;
        depth=40,
        nheads=32,
        ratio=3,
        kwargs...
    )
    return model
end
