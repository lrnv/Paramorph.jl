using Test
using Paramorph

const ValidationTV = Paramorph.TransformVariables

@paramorph T struct ValidatedChild{T<:Real}
    order::T ~ ValidationTV.asℝ₊
end

@paramorph T struct GloballyValidated{D,T<:Real,C<:ValidatedChild}
    child::C ~ nested()
    weights::Vector{T} ~ ValidationTV.as(Vector, ValidationTV.asℝ₊, D)
    @validate child.order < sum(weights)
end

@paramorph T struct ContextValidated{T<:Real}
    x::T ~ ValidationTV.asℝ
    @validate x <= get(context, :upper, Inf)
end

@paramorph T struct ThrowingValidation{T<:Real}
    x::T ~ ValidationTV.asℝ
    @validate x >= 0 || throw(ArgumentError("x must be nonnegative"))
end

@paramorph T struct InvalidValidationResult{T<:Real}
    x::T ~ ValidationTV.asℝ
    @validate x
end

const LooseValidatedChild = ValidatedChild{T} where T

struct UnionAllHolder{C}
    child::C
end
const LooseHeldChild = fieldtype(UnionAllHolder{C} where C<:LooseValidatedChild, :child)

@testset "whole-object validation" begin
    child = ValidatedChild(1.0)
    valid = GloballyValidated{2,Float64,typeof(child)}(child, [1.0, 1.0])
    @test valid.weights == [1.0, 1.0]
    @test_throws DomainError GloballyValidated{2,Float64,typeof(child)}(
        ValidatedChild(3.0),
        [1.0, 1.0],
    )

    rebuilt = constraint(valid, zeros(3))
    @test rebuilt.child.order ≈ 1.0
    @test rebuilt.weights == [1.0, 1.0]
    @test_throws DomainError constraint(valid, [log(3.0), 0.0, 0.0])

    @test ContextValidated(2.0).x == 2.0
    @test constraint(ContextValidated{Float64}, [2.0]; context=(; upper=3.0)).x == 2.0
    @test_throws DomainError constraint(
        ContextValidated{Float64},
        [2.0];
        context=(; upper=1.0),
    )

    @test_throws ArgumentError ThrowingValidation(-1.0)
    @test_throws ArgumentError InvalidValidationResult(1.0)
end

@testset "unbounded UnionAll numeric rebinding" begin
    prototype = parameter_prototype(LooseValidatedChild; numeric_type=Float32)
    @test prototype isa ValidatedChild{Float32}
    @test prototype.order ≈ 1.0f0

    held = parameter_prototype(LooseHeldChild; numeric_type=Float32)
    @test held isa ValidatedChild{Float32}

    @test Paramorph.rebind_numeric_type(LooseValidatedChild, Float32) ===
          ValidatedChild{Float32}
end
