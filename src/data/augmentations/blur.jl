struct GaussianBlur{S <: Distributions.Sampleable} <: DataAugmentation.Transform
	dist::S
	l::Int
end

GaussianBlur(min = 0.1, max = 2.0) = GaussianBlur(Distributions.Uniform(min, max), 9)

DataAugmentation.getrandstate(transform::GaussianBlur) = rand(transform.dist)

function DataAugmentation.apply(transform::GaussianBlur, item::Image; randstate = DataAugmentation.getrandstate(transform))
	σ = randstate
	return DataAugmentation.setdata(item, gaussianblur(DataAugmentation.itemdata(item), σ, transform.l))
end

function DataAugmentation.apply!(buffer, transform::GaussianBlur, item::Image; randstate = DataAugmentation.getrandstate(transform))
	σ = randstate
	gaussianblur!(DataAugmentation.itemdata(buffer), DataAugmentation.itemdata(item), σ, transform.l)
	return buffer
end

function gaussianblur(image, σ, l)
	return gaussianblur!(copy(image), image, σ, l)
end

function gaussianblur!(dst::AbstractArray{DST}, src::AbstractArray{SRC}, σ, l) where {DST, SRC}
	kernel = ImageFiltering.Kernel.gaussian((σ, σ), (l, l))
	imfilter!(
		dst,
		src,
		kernel
	)
	return dst
end
