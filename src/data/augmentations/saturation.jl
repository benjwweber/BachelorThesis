struct AdjustSaturation{S<:Distributions.Sampleable} <: DataAugmentation.Transform
	dist::S
	clamp::Bool
end

AdjustSaturation(f::Real; clamp::Bool=true) = AdjustSaturation(Uniform(max(0, 1 - f), 1 + f), clamp)

DataAugmentation.getrandstate(transform::AdjustSaturation) = rand(transform.dist)

function DataAugmentation.apply(transform::AdjustSaturation, item::Image; randstate = DataAugmentation.getrandstate(transform))
	ratio = randstate
	return DataAugmentation.setdata(item, adjustsaturation(DataAugmentation.itemdata(item), ratio, transform.clamp))
end

function DataAugmentation.apply!(buffer, transform::AdjustSaturation, item::Image; randstate = DataAugmentation.getrandstate(transform))
	ratio = randstate
	adjustsaturation!(DataAugmentation.itemdata(buffer), DataAugmentation.itemdata(item), ratio, transform.clamp)
	return buffer
end

function adjustsaturation(image, ratio, clamp)
	return adjustsaturation!(copy(image), ratio, clamp)
end

function adjustsaturation!(dst::AbstractArray{U}, src::AbstractArray{T}, ratio, clamp) where {T, U}
	map!(dst, src) do x
		x = ratio * x + (1 - ratio) * Colors.Gray(x)
		x = clamp ? Colors.mapc(Colors.clamp01, x) : x
		return convert(U, x)
	end
end

adjustsaturation!(image, ratio, clamp) = adjustsaturation!(image, image, ratio, clamp)
