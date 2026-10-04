@concrete struct GramLoss
	image
	normalize
	clamp0_student
	clamp0_teacher
	mseloss
end

function GramLoss(;
	image = false,
	normalize::Bool,
	clamp0_student::Bool,
	clamp0_teacher::Bool,
	kwargs_mseloss
)
	return GramLoss(
		Val(image),
		normalize ? normalize2_1 : identity,
		clamp0_student ? clamp0 : identity,
		clamp0_teacher ? clamp0 : identity,
		MSELoss(;kwargs_mseloss...)
	)
end

# const mseloss = Lux.MSELoss(;agg = mean)

normalize2_1(x) = normalize2(x; dims = 1, epsilon = convert(Lux.unwrapped_eltype(x), 1f-12))
clamp0(x) = max.(x, 0)

function (loss::GramLoss)(features_student, ::Nothing)
	return 0
end

function (gramloss::GramLoss{Val{false}})(
	features_student,
	features_teacher
)
	features_student_flatten = flatten_tail(features_student)
	features_teacher_flatten = flatten_tail(features_teacher)
	features_teacher_normalize = gramloss.normalize(features_teacher)
	features_student_normalize = gramloss.normalize(features_student)

	features_teacher_matmul = features_teacher_normalize' * features_teacher_normalize
	features_student_matmul = features_student_normalize' * features_student_normalize

	return gramloss.mseloss(gramloss.clamp0_student(features_student_gram), gramloss.clamp0_teacher(features_teacher_gram))
end

function (gramloss::GramLoss{Val{true}})(
	features_student,
	features_teacher
)
	features_teacher_normalize = gramloss.normalize(features_teacher)
	features_student_normalize = gramloss.normalize(features_student)

	features_teacher_matmul = Lux.LuxLib.API.batched_matmul(
		features_teacher_normalize,
		features_teacher_normalize;
		lhs_contracting_dim = 1,
	)
	features_student_matmul = Lux.LuxLib.API.batched_matmul(
		features_student_normalize,
		features_student_normalize;
		lhs_contracting_dim = 1
	)
	return gramloss.mse(gramloss.clamp0_student(features_student_gram), gramloss.clamp0_teacher(features_teacher_gram))
end
