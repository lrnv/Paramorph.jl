using Aqua
using Paramorph
using Test

const TV = Paramorph.TransformVariables

@paramorph T struct PositiveScalarParameter{T<:Real}
    scale::T ~ TV.asℝ₊
    label::String = "default"
end

@paramorph T struct RuntimeSizedSimplex{T<:Real}
    n::Int
    weights::Vector{T} ~ TV.UnitSimplex(n)
end

opaque_vector_geometry(n, ::Type{T}) where {T} =
    TV.as(Vector, bounded_interval(-one(T), one(T)), n)

@paramorph T struct OpaqueVectorGeometry{T<:Real}
    n::Int
    θ::Vector{T} ~ opaque_vector_geometry(n, T)
end

@paramorph T struct ContextBoundScalar{T<:Real}
    θ::T ~ closed_lower(get(context, :lower, -one(T)))
end

@paramorph T struct NestedContextContainer{D,T<:Real}
    child::ContextBoundScalar{T} ~ nested(lower=-inv(T(D - 1)))
end

@paramorph T struct PositiveLeaf{T<:Real}
    x::T ~ TV.asℝ₊
end

@paramorph T struct RecursivePositiveCollection{N,T<:Real}
    children::Vector{PositiveLeaf{T}} ~ recursive(N)
end

@paramorph T struct RuntimeCompositeGeometry{T<:Real}
    d::Int
    radial::Vector{T} ~ TV.as(Vector, closed_lower(one(T)), 2^d - d - 1)
    simplices::Vector{Vector{T}} ~ repeat_transform(TV.UnitSimplex(2^(d - 1)), d)
end

function ordered_pair_transform()
    base = TV.as((lower=TV.asℝ, gap=TV.asℝ₊))
    forward = p -> (; lower=p.lower, upper=p.lower + p.gap)
    backward = p -> (; lower=p.lower, gap=p.upper - p.lower)
    return joint_transform(base, forward, backward)
end

@paramorph T struct CoupledOrderedPair{T<:Real}
    lower::T
    upper::T
    @geometry ((lower, upper) ~ ordered_pair_transform())
end

@paramorph T struct UnitIntervalParameter{T<:Real}
    p::T ~ bounded_interval(zero(T), one(T))
end

@paramorph T struct StrictPositiveParameter{T<:Real}
    x::T ~ open_lower(zero(T))
end


@testset "Paramorph" begin
    @testset "storage and geometry are separate" begin
        x = PositiveScalarParameter{Float64}(2.0, "custom")
        @test Paramorph.parameter_fields(typeof(x)) == (:scale,)
        @test Paramorph.auxiliary_fields(typeof(x)) == (:label,)
        @test intrinsic_dimension(x) == 1
        @test unconstrain(constraint(x, unconstrain(x))) ≈ unconstrain(x)

        from_type = constraint(PositiveScalarParameter{Float64}, [0.0])
        @test from_type.scale ≈ 1.0
        @test from_type.label == "default"

        rebound = constraint(PositiveScalarParameter{Float64}, Float32[0.0])
        @test rebound isa PositiveScalarParameter{Float32}

        y, logjac = constraint_with_logjac(PositiveScalarParameter{Float64}, [0.0])
        @test y.scale ≈ 1.0
        @test logjac ≈ 0.0
    end

    @testset "runtime-sized simplex geometry" begin
        x = RuntimeSizedSimplex{Float64}(3, [0.2, 0.3, 0.5])
        @test intrinsic_dimension(x) == 2
        @test_throws ArgumentError intrinsic_dimension(RuntimeSizedSimplex{Float64})
        @test_throws ArgumentError constraint(RuntimeSizedSimplex{Float64}, zeros(2))

        @test intrinsic_dimension(
            RuntimeSizedSimplex{Float64}; auxiliary=(; n=3),
        ) == 2
        typed = constraint(
            RuntimeSizedSimplex{Float64}, zeros(2); auxiliary=(; n=3),
        )
        @test typed.n == 3
        @test length(typed.weights) == 3
        @test sum(typed.weights) ≈ 1.0

        y = constraint(x, zeros(2))
        @test y.n == 3
        @test length(y.weights) == 3
        @test sum(y.weights) ≈ 1.0
    end

    @testset "opaque runtime vector geometry" begin
        x = OpaqueVectorGeometry{Float64}(3, [0.2, -0.4, 0.8])
        @test intrinsic_dimension(x) == 3
        @test_throws ArgumentError intrinsic_dimension(OpaqueVectorGeometry{Float64})
        @test intrinsic_dimension(
            OpaqueVectorGeometry{Float64}; auxiliary=(; n=3),
        ) == 3
        typed = constraint(
            OpaqueVectorGeometry{Float64}, zeros(3); auxiliary=(; n=3),
        )
        @test typed.n == 3
        @test all(iszero, typed.θ)
        @test unconstrain(constraint(x, unconstrain(x))) ≈ unconstrain(x)
        @test all(abs.(constraint(x, zeros(3)).θ) .<= 1)
    end

    @testset "nested context geometry" begin
        p = NestedContextContainer{3}(ContextBoundScalar(0.2))
        @test intrinsic_dimension(NestedContextContainer{3,Float64}) == 1
        q = constraint(NestedContextContainer{3,Float64}, [0.0])
        @test q isa NestedContextContainer{3,Float64}
        @test q.child isa ContextBoundScalar{Float64}
        @test q.child.θ ≈ 0.5
        @test unconstrain(constraint(p, unconstrain(p))) ≈ unconstrain(p)
    end

    @testset "recursive nested geometry" begin
        f = RecursivePositiveCollection{2}([PositiveLeaf(1.0), PositiveLeaf(2.0)])
        @test intrinsic_dimension(RecursivePositiveCollection{2,Float64}) == 2
        rebuilt = constraint(f, unconstrain(f))
        @test [leaf.x for leaf in rebuilt.children] ≈ [1.0, 2.0]
    end

    @testset "runtime composite geometry" begin
        x = RuntimeCompositeGeometry{Float64}(
            2,
            [1.5],
            [[0.4, 0.6], [0.7, 0.3]],
        )
        @test intrinsic_dimension(x) == 3
        @test_throws ArgumentError intrinsic_dimension(RuntimeCompositeGeometry{Float64})
        @test intrinsic_dimension(
            RuntimeCompositeGeometry{Float64}; auxiliary=(; d=2),
        ) == 3
        typed = constraint(
            RuntimeCompositeGeometry{Float64}, zeros(3); auxiliary=(; d=2),
        )
        @test typed.d == 2
        @test length(typed.radial) == 1
        @test length(typed.simplices) == 2
        @test all(w -> sum(w) ≈ 1.0, typed.simplices)

        rebuilt = constraint(x, zeros(3))
        @test rebuilt.d == 2
        @test length(rebuilt.radial) == 1
        @test length(rebuilt.simplices) == 2
        @test all(w -> sum(w) ≈ 1.0, rebuilt.simplices)
    end

    @testset "global coupled geometry" begin
        pair = CoupledOrderedPair(0.0, 2.0)
        @test Paramorph.parameter_fields(typeof(pair)) == (:lower, :upper)
        @test Paramorph.auxiliary_fields(typeof(pair)) == ()
        @test intrinsic_dimension(CoupledOrderedPair{Float64}) == 2
        @test_throws DomainError CoupledOrderedPair(2.0, 1.0)

        zero_pair = constraint(CoupledOrderedPair{Float64}, zeros(2))
        @test zero_pair.lower ≈ 0.0
        @test zero_pair.upper ≈ 1.0
    end

    @testset "constructor validation" begin
        @test UnitIntervalParameter(0.0).p == 0.0
        @test UnitIntervalParameter(1.0).p == 1.0
        @test_throws DomainError UnitIntervalParameter(-0.1)
        @test_throws DomainError UnitIntervalParameter(1.1)
        @test StrictPositiveParameter(1.0).x == 1.0
        @test_throws DomainError StrictPositiveParameter(0.0)
    end
end

include("matrix_geometries.jl")

Aqua.test_all(Paramorph)