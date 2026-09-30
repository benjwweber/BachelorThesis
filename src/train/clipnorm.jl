using Functors
using Accessors
using AccessorsExtra

function norm2(gradients::NamedTuple)
    return sqrt(sum(fmap(Base.Fix1(sum, abs2), gradients; walk=Functors.IterateWalk())))
end

function clipnorm(gradients, omega)
    nrm = norm2(gradients)
    λ = min(omega / nrm, 1)
    return fmap(Base.Fix2(*, λ), gradients) # why not Fix1
end

function clipnorm_backbone_heads(gradients, omega)
    return modify(
        Base.Fix2(clipnorm, omega),
        gradients,
        @optic _.backbone _.head_dino _.head_ibot
    )
end



