module Losses
	using Lux
	using LinearAlgebra
	using Statistics
	using Functors
	using Random
	using ConcreteStructs

	export
		DINOLoss,
		GramLoss,
		iBOTLoss,
		KoLeoLoss,
		PreTrainingObjective

	include("../apply.jl")
	include("../container.jl")
	include("../utils.jl")

	include("./dino.jl")
	include("./gram.jl")
	include("./ibot.jl")
	include("./koleo.jl")
	include("./pre.jl")
	# include("./ref.jl")
end
