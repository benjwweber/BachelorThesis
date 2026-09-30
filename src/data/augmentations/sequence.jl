struct RandomSequence{T<:Tuple} <: DataAugmentation.Transform
	transforms::T
end

RandomSequence(transforms...) = RandomSequence{typeof(transforms)}(transforms)
RandomSequence(transform::DataAugmentation.Transform) = transform

DataAugmentation.getrandstate(sequence::RandomSequence) = (
	randperm(length(sequence.transforms)), DataAugmentation.getrandstate.(sequence.transforms)
)

function DataAugmentation.apply(sequence::RandomSequence, items; randstate = DataAugmentation.getrandstate(sequence))
	for (transforms, randstate) in zip(sequence.transforms[randstate[1]], randstate[2][randstate[1]])
		items = DataAugmentation.apply(transforms, items; randstate)
	end
	return items
end

function DataAugmentation.makebuffer(sequence::RandomSequence, items)
	buffers = []
	for transforms in sequence.transforms
		push!(buffers, DataAugmentation.makebuffer(transforms, items))
	end
	return buffers
end


function DataAugmentation.apply!(buffers, sequence::RandomSequence, items; randstate = getrandstate(sequence))
	#@assert length(buffers) == length(sequence.transforms)
	for (transforms, buffer, randstate) in zip(
		sequence.transforms[randstate[1]],
		buffers[randstate[1]],
		randstate[2][randstate[1]]
	)
		items = apply!(buffer, transforms, items; randstate)
	end
	return items
end
