using Functors

function accumulate!(accumulator, dx, n)
    @. accumulator += dx / n
end

function accumulate!(accumulator::NamedTuple, dx::NamedTuple, n)
    fmap(
        Base.Fix{3}(accumulate!, n),
        accumulator,
        dx
    )
    return accumulator
end

function decumulate!(accumulator)
    accumulator .= zero(accumulator)
end
function decumulate!(accumulator::NamedTuple)
    fmap(decumulate!, accumulator)
    return accumulator
end
