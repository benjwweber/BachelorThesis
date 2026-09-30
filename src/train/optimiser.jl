using Accessors
using AccessorsExtra
using Functors
using Lux
using NamedTupleTools

function nthblock(keypath, nblocks)
    m = match(r"block_(?<index>\d+)", Functors.keypathstr(keypath))
    if !isnothing(m)
        return parse(Int, m["index"])
    end
    if :backbone in keypath
        return 0
    end
    return nblocks + 1
end

function compute_factors_eta(parameters, f; embedding_patch)
    nblocks = length(parameters.backbone.blocks)
    factors = fmap_with_path(parameters) do keypath, value
        factor = f
        factor ^= (nblocks + 1 - nthblock(keypath, nblocks))
        :embedding_patch ∈ keypath && return factor * embedding_patch
        return factor
    end
    factors_eta_backbone = Accessors.delete(
        Accessors.delete(
            factors,
            @optics(_.head_dino, _.head_ibot)
        ),
        @optic(_.embedding_patch)
    )
    factors_eta_heads = Accessors.delete(
        Accessors.delete(
            factors,
            @optic(_.backbone)
        ),
        @optic(_.head_dino.dense_out, _.head_ibot.dense_out)
    )
    factors_eta_heads = merge((; backbone=(; embedding_patch=factors.backbone.embedding_patch)), factors_eta_heads)
    factors_eta_lasts = Accessors.delete(
        Accessors.delete(
            factors,
            @optic(_.backbone)
        ),
        @optic(_.head_dino.dense_in, _.head_dino.dense_bottleneck, _.head_ibot.dense_in, _.head_ibot.dense_bottleneck)
    )
    return (;
        backbone=factors_eta_backbone,
        heads=factors_eta_heads,
        lasts=factors_eta_lasts
    )
end

function compute_etas(etas, factors)
    etas_backbone = fmap(Base.Fix1(*, etas.backbone), factors.backbone)
    etas_heads = fmap(Base.Fix1(*, etas.heads), factors.heads)
    etas_lasts = fmap(Base.Fix1(*, etas.lasts), factors.lasts)
    etas = merge_recursive(etas_backbone, etas_heads, etas_lasts)
    return etas
end

function compute_etas_lambdas(
    etas,
    factors_etas,
    lambda,
    factors_lambda
)
    return (; etas=compute_etas(etas, factors_etas), lambdas=compute_lambdas(lambda, factors_lambda))
end

function compute_factors_lambda(parameters, factor; head_dino=1.0f0)
    lambdas = fmap_with_path(parameters) do keypath, value
        any((:bias, :scale) .∈ Ref(keypath)) && return 0.0f0
        :head_dino ∈ keypath && return factor * head_dino
        return factor
    end
    return lambdas
end

function compute_lambdas(
    lambda, factors
)
    lambdas = fmap(Base.Fix1(*, lambda), factors)
    return lambdas
end


function Optimisers.adjust!(student::Lux.Training.TrainState; etas::NamedTuple)
    fmap(adjust_eta!, student.optimizer_state, etas; exclude=Base.Fix2(isa, Optimisers.Leaf))
    return student
end

function adjust_eta!(leaf::Optimisers.Leaf, eta::Reactant.ConcreteRNumber)
    leaf.rule = set(leaf.rule, @optic(_.opt.eta), eta)
    return nothing
end

function Optimisers.adjust!(student::Lux.Training.TrainState; lambdas::NamedTuple)
    fmap(adjust_lambda!, student.optimizer_state, etas; exclude=Base.Fix2(isa, Optimisers.Leaf))
    return student
end

function adjust_lambda!(leaf::Optimisers.Leaf, lambda::Reactant.ConcreteRNumber)
    leaf.rule = set(leaf.rule, @optic(_.opt.lambda), lambda)
    return nothing
end

function Optimisers.adjust!(student::Lux.Training.TrainState; etas::NamedTuple, lambdas::NamedTuple)
    fmap(adjust_eta_lambda!, student.optimizer_state, etas, lambdas; exclude=Base.Fix2(isa, Optimisers.Leaf))
    return student
end

function adjust_eta_lambda!(leaf::Optimisers.Leaf, eta::Reactant.ConcreteRNumber, lambda::Reactant.ConcreteRNumber)
    leaf.rule = setall(leaf.rule, @optics(_.opt.eta, _.opt.lambda), (eta, lambda))
    return nothing
end
