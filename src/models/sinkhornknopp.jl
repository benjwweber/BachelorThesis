function sinkhornknopp(
    inputs, # |> size == (Color, Batches...)
    temperature=1;
    niterations=3
)
    # return inputs
    T = Lux.unwrapped_eltype(inputs)
    Q = @. exp(inputs / temperature)
    (ncolors, temp...) = size(Q)
    nbatches = prod(temp)
    dim_color = 1
    dims_batches = 2:ndims(inputs)
    sum_Q = sum(Q)
    @. Q = Q / sum_Q
    for _ in 1:niterations
        # Q = @. Q / $sum(Q; dims=dims_batches) / ncolors
        sum_batch = sum(Q; dims=dims_batches)
        @. Q = Q / sum_batch / ncolors

        sum_color = sum(Q; dims=dim_color)
        @. Q = Q / sum_color / nbatches
    end
    @. Q = Q * nbatches
    return Q # |> size == (Color, Batches...)
end

function sinkhornknopp(
    inputs, # |> size == (Color, Batches...)
    masks,
    temperature;
    niterations=3
)
    T = Lux.unwrapped_eltype(inputs)
    Q = @. exp(inputs / temperature)
    Q .= LuxLib.Impl.dropout_dot_mul(Q, masks) # inputs .* masks

    ncolors = size(Q, 1)
    nbatches = sum(masks)
    dim_color = 1
    dims_batches = 2:ndims(inputs)
    sum_Q = sum(Q)
    @. Q = Q / sum_Q
    epsilon = convert(T, 1.0f-12)
    for _ in 1:niterations
        sum_batch = sum(Q; dims=dims_batches)
        @. Q = Q / max(sum_batch, epsilon) / ncolors # maybe think of better solution to not divide by 0

        sum_color = sum(Q; dims=dim_color)
        @. Q = Q / max(sum_color, epsilon) / nbatches
    end
    @. Q = Q * nbatches
    return Q # |> size == (Color, Batches...)
end

using NNlib: logsumexp

function sinkhornknopp_logspace(
    inputs,
    masks,
    temperature;
    niterations=3
)
    T = Lux.unwrapped_eltype(inputs)

    # Map valid positions to (inputs / temperature) and masked entries to -Inf
    m = typemin(T)
    log_Q = @. ifelse(masks, inputs / temperature, m)

    (ncolors, temp...) = size(log_Q)
    nbatches = sum(masks)
    dims_color = 1
    dims_batches = 2:ndims(inputs)

    # Initial log-normalization: log_Q .- log(sum(Q))
    log_sum_Q = logsumexp(log_Q)
    @. log_Q = log_Q - log_sum_Q

    log_ncolors = log(convert(T, ncolors))
    log_nbatches = log(one(T) * nbatches)

    for _ in 1:niterations
        # Normalize over batch dimensions: subtract log(sum(Q, dims=dims_batches))
        log_sum_batch = logsumexp(log_Q; dims=dims_batches)
        @. log_Q = log_Q - log_sum_batch - log_ncolors

        # Normalize over color dimension: subtract log(sum(Q, dims=dim_color))
        log_sum_color = LuxLib.Impl.dropout_dot_mul(logsumexp(log_Q; dims=dims_color), masks)
        @. log_Q = log_Q - log_sum_color - log_nbatches
    end

    # Return normalized assignment probabilities in linear domain
    # exp(-Inf) naturally evaluates to 0.0
    Q = @. exp(ifelse(masks, log_Q + log_nbatches, m))
    return LuxLib.Impl.dropout_dot_mul(Q, masks)
end

function sinkhornknopp_logspace(
    inputs,
    temperature;
    niterations=3
)
    T = Lux.unwrapped_eltype(inputs)

    # Map valid positions to (inputs / temperature) and masked entries to -Inf
    log_Q = inputs ./ temperature

    (ncolors, temp...) = size(log_Q)
    nbatches = prod(temp)
    dims_color = 1
    dims_batches = 2:ndims(inputs)

    # Initial log-normalization: log_Q .- log(sum(Q))
    log_sum_Q = logsumexp(log_Q)
    @. log_Q = log_Q - log_sum_Q

    log_ncolors = log(convert(T, ncolors))
    log_nbatches = log(convert(T, nbatches))

    for _ in 1:niterations
        # Normalize over batch dimensions: subtract log(sum(Q, dims=dims_batches))
        log_sum_batch = logsumexp(log_Q; dims=dims_batches)
        @. log_Q = log_Q - log_sum_batch - log_ncolors

        # Normalize over color dimension: subtract log(sum(Q, dims=dim_color))
        log_sum_color = logsumexp(log_Q; dims=dims_color)
        @. log_Q = log_Q - log_sum_color - log_nbatches

        # # Explicitly enforce -Inf on masked entries to prevent floating-point drift
        # log_Q = ifelse.(masks .== 1, log_Q, T(-Inf))
    end

    # Return normalized assignment probabilities in linear domain
    # exp(-Inf) naturally evaluates to 0.0
    return @. exp(log_Q + log_nbatches)
end
