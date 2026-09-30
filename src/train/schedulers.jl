using ParameterSchedulers

struct DeviceScheduler{T}
    schedule
    device
end

function (schedule::DeviceScheduler{T})(iteration) where {T}
    return schedule.device(T(schedule.schedule(iteration)))
end

function TriangleCosAnnealConstant(;
    start,
    peak,
    stop,
    freeze=0,
    niterations=(;
        freeze=0,
        warmup=0,
        cosine=0
    ),
    eltype,
    device
)

    cos_anneal = CosAnneal(l0=peak, l1=stop, period=niterations.cosine)
    if stop > peak
        cos_anneal = ParameterSchedulers.reverse(cos_anneal, niterations.cosine + 1)
    end
    return DeviceScheduler{eltype}(
        ParameterSchedulers.Sequence(
            ParameterSchedulers.Constant(freeze) => niterations.freeze,
            Triangle(l0=start, l1=peak, period=2 * niterations.warmup) => niterations.warmup,
            cos_anneal => niterations.cosine,
            ParameterSchedulers.Constant(stop) => 1
        ),
        device
    )
end
