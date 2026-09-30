struct AdjustHue{S<:Distributions.Sampleable} <: DataAugmentation.Transform
	dist::S
end

function AdjustHue(f::Real)
	#@argcheck 0 < f <= 180
	return AdjustHue(Uniform(-f, f))
end

DataAugmentation.getrandstate(transform::AdjustHue) = rand(transform.dist)

function DataAugmentation.apply(
	transform::AdjustHue,
	item::Image;
	randstate = DataAugmentation.getrandstate(transform)
)
	factor = randstate
	return DataAugmentation.setdata(item, adjusthue(DataAugmentation.itemdata(item), factor))
end

function DataAugmentation.apply!(buffer, transform::AdjustHue, item::Image; randstate = DataAugmentation.getrandstate(transform))
	factor = randstate
	adjusthue!(DataAugmentation.itemdata(buffer), DataAugmentation.itemdata(item), factor)
	return buffer
end

function adjusthue(img, factor)
	return adjusthue!(copy(img), factor)
end

adjusthue!(img, factor) = adjusthue!(img, img, factor)

function adjusthue!(destination::AbstractArray{U}, source::AbstractArray{T}, factor) where {T, U}
	map!(destination, source) do pixel
		hsv = HSV(pixel)
		h = Colors.normalize_hue(hue(hsv) + factor)
		#h = mod(h + factor), 360)
		convert(U, HSV(h, hsv.s, hsv.v))
	end
end
