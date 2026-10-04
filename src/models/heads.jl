@containerlayer struct DINOHead
    dense_in
    chain
    dense_bottleneck
    normalize
    dense_out
end

function DINOHead(
    (dim_in, (dim_hidden, (dim_bottleneck, dim_out))),
    activation=Lux.NNlib.gelu;
    nlayers=4,
    epsilon=1f-12,
    use_batchnorm=false,
    kwargs_dense=(;
        use_bias=true,
        init_weight=Lux.WeightInitializers.truncated_normal(Float32; std=0.02f0),
        init_bias=zeros32
    ),
    kwargs_dense_out=(;
        init_weight=Lux.WeightInitializers.truncated_normal(Float32; std=0.02f0),
        init_bias=zeros32
    ),
    kwargs_batchnorm=(;),
)
    activation_dense = use_batchnorm ? identity : activation
    if nlayers == 2
        dense_in = Lux.Dense(
            dim_in => dim_bottleneck;
            kwargs_dense...
        )
        chain = Lux.NoOpLayer()
        dense_bottleneck = Lux.NoOpLayer()
    else
        dense_in = Lux.Dense(
            dim_in => dim_hidden,
            activation_dense;
            kwargs_dense...
        )
        nlayers_chain = nlayers - 3
        if nlayers_chain == 0
            chain = Lux.NoOpLayer()
        else
            layers = []
            for i in 1:nlayers_chain
                if use_batchnorm
                    push!(
                        layers,
                        Symbol(:batchnorm_, i) => Lux.BatchNorm(
                            dim_hidden,
                            activation;
                            kwargs_batchnorm...
                        )
                    )
                end
                push!(
                    layers,
                    Symbol(:dense_, i) => Lux.Dense(
                        dim_hidden => dim_hidden,
                        activation_dense;
                        kwargs_dense...
                    )
                )
            end
            if use_batchnorm
                push!(
                    layers,
                    Symbol(:batchnorm_, nlayers_chain + 1) => Lux.BatchNorm(
                        dim_hidden,
                        activation;
                        kwargs_batchnorm...
                    )
                )
            end
            chain = Lux.Chain(; layers...)
        end
        dense_bottleneck = Lux.Dense(
            dim_hidden => dim_bottleneck;
            kwargs_dense...
        )
    end
    normalize = Normalize2(1; epsilon)
    dense_out = Lux.Dense(
        dim_bottleneck => dim_out;
        merge(kwargs_dense_out, (; use_bias=false))...
    )
    return DINOHead(
        dense_in,
        chain,
        dense_bottleneck,
        normalize,
        dense_out
    )
end

# this could be a chain loool
@apply function (head::DINOHead)(inputs)
    return inputs |>
           head.dense_in |>
           head.chain |>
           head.dense_bottleneck |>
           head.normalize |>
           head.dense_out
end
