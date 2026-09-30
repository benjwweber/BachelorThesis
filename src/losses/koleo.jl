function pairwisedistance2(x, y; dims=1, epsilon=1f-8)
    return norm2(@.(x - y + epsilon); dims)
end

function Base.mod1(x, y)
    m = mod(x, y)
    return ifelse(m == 0, y, m)
end

torowindices(indices, d) = mod1.(indices, Int(size(d, 1)))
#torowindices(indices::Vector{CartesianIndex}, d) = Base.Fix2(getindex, 1).(indices)

function pairwisenearestneighborsinner(x)
    d = x' * x
    view(d, diagind(d)) .= -1
    # dd = d - Diagonal(d)
    indices = findmax(d; dims=1)[2]
    indices_mod = mod.(indices, Int(size(d, 1)))
    indices_mod1 = ifelse.(indices_mod .== 0, Int(size(d, 1)), indices_mod)
    return dropdims(indices_mod1; dims=1)
    # indices_rows = torowindices(indices, size(d, 1))
    # return dropdims(indices_rows; dims = 1)
end

struct KoLeoLoss
    epsilon
end
KoLeoLoss(; epsilon) = KoLeoLoss(epsilon)

function (koleoloss::KoLeoLoss)(
    classes_student # |> size == (Color, Crop, Batch)
)
    return compute_koleoloss(
        classes_student;
        epsilon=convert(Lux.unwrapped_eltype(classes_student), koleoloss.epsilon)
    )
end

function compute_koleoloss(
    classes_student::AbstractArray{T,2}; # |> size == (Color, Batch)
    epsilon=1f-8
) where {T}
    return compute_koleoloss_(classes_student, epsilon)
end

function compute_koleoloss_test(
    classes_student::AbstractArray{T,3}; # |> size == (Color, Crop, Batch)
    epsilon=1f-8
) where {T}

    # return sum(compute_koleoloss_, eachslice(x, dims = 2))
    # return sum(mapslices(compute_koleoloss_, classes_student; dims = (1, 3)))
    return sum(mapslices(Base.Fix2(compute_koleoloss_, epsilon), classes_student; dims=(1, 3)))
end

function compute_koleoloss_(classes_student, epsilon=1f-8)
    classes_normalize = normalize2(classes_student; epsilon, dims=1)
    indices = pairwisenearestneighborsinner(classes_normalize)
    classes_nearestneighbor = Lux.NNlib.gather(classes_normalize, indices)
    distances = pairwisedistance2(classes_normalize, classes_nearestneighbor; dims=1, epsilon)
    loss = -mean(log, distances .+ epsilon)
    return loss
end
