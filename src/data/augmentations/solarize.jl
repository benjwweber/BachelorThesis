struct Solarize <: DataAugmentation.Transform
	threshold
end

function DataAugmentation.apply(transform::Solarize, item::Image; randstate = nothing)
	return DataAugmentation.setdata(item, solarize(DataAugmentation.itemdata(item), transform.threshold))
end

function DataAugmentation.apply!(buffer, transform::Solarize, item::Image; randstate = nothing)
	solarize!(DataAugmentation.itemdata(buffer), DataAugmentation.itemdata(item), transform.threshold)
	return buffer
end

function solarize(image, threshold)
	return solarize!(copy(image), threshold)
end

solarize!(image, threshold) = solarize!(image, image, threshold)

function solarize!(dst::AbstractArray{DST}, src::AbstractArray{SRC}, threshold) where {DST, SRC}
	map!(dst, src) do rgb
		convert(DST, Colors.mapc(Base.Fix2(_solarize, threshold), rgb))
	end
end

function _solarize(value, threshold)
	if value <= threshold
		return value
	end
	return Colors.clamp01(1 - value)
end
