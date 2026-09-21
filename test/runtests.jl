using Aqua
using Paramorph
using Test

const TV = Paramorph.TransformVariables

@testset "Paramorph" begin
    @testset "storage and geometry are separate" begin
        @paramorph T struct PositiveScale{T<:Real}
            scale::T ~ TV.asℝ₊
            label::String = "default"
        end

        x = PositiveScale(2.0, "custom")
        @test Paramorph.parameter_fields(typeof(x)) == (:scale,)
        @test Paramorph.auxiliary_fields(typeof(x)) == (:label,)
        @test intrinsic_dimension(x) == 1
        @test unconstrain(constraint(x, unconstrain(x))) ≈ unconstrain(x)

        from_type = constraint(PositiveScale{Float64}, [0.0])
        @test from_type.scale ≈ 1.0
        @test from_type.label == "default"

        rebound = constraint(PositiveScale{Float64}, Float32[0.0])
        @test rebound isa PositiveScale{Float32}

        y, logjac = constraint_with_logjac(PositiveScale{Float64}, [0.0])
        @test y.scale ≈ 1.0
        @test logjac ≈ 0.0
    end

    @testset "runtime field dependent geometry" begin
        @paramorph T struct DynamicSimplex{T<:Real}
            n::Int
            weights::Vector{T} ~ TV.UnitSimplex(n)
        end

        x = DynamicSimplex(3, [0.2, 0.3, 0.5])
        @test intrinsic_dimension(x) == 2
        @test_throws ArgumentError intrinsic_dimension(DynamicSimplex{Float64})
        @test_throws ArgumentError constraint(DynamicSimplex{Float64}, zeros(2))

        y = constraint(x, zeros(2))
        @test y.n == 3
        @test length(y.weights) == 3
        @test sum(y.weights) ≈ 1.0
    end

    @testset "nested context" begin
        @paramorph T struct Child{T<:Real}
            θ::T ~ closed_lower(get(context, :lower, -one(T)))
        end

        @paramorph T struct Parent{D,T<:Real}
            child::Child{T} ~ nested(lower=-inv(T(D - 1)))
        end

        p = Parent{3}(Child(0.2))
        @test intrinsic_dimension(Parent{3,Float64}) == 1
        q = constraint(Parent{3,Float64}, [0.0])
        @test q isa Parent{3,Float64}
        @test q.child isa Child{Float64}
        @test q.child.θ ≈ 0.5
        @test unconstrain(constraint(p, unconstrain(p))) ≈ unconstrain(p)
    end

    @testset "recursive nested structures" begin
        @paramorph T struct Leaf{T<:Real}
            x::T ~ TV.asℝ₊
        end

        @paramorph T struct Forest{N,T<:Real}
            children::Vector{Leaf{T}} ~ recursive(N)
        end

        f = Forest{2}([Leaf(1.0), Leaf(2.0)])
        @test intrinsic_dimension(Forest{2,Float64}) == 2
        rebuilt = constraint(f, unconstrain(f))
        @test [leaf.x for leaf in rebuilt.children] ≈ [1.0, 2.0]
    end

    @testset "Copulas-style value dependent declarations" begin
        @paramorph T struct TawnLike{T<:Real}
            d::Int
            dep::Vector{T} ~ TV.as(Vector, closed_lower(one(T)), 2^d - d - 1)
            weights::Vector{Vector{T}} ~ repeat_transform(TV.UnitSimplex(2^(d - 1)), d)
        end

        t = TawnLike(
            2,
            [1.5],
            [[0.4, 0.6], [0.7, 0.3]],
        )
        @test intrinsic_dimension(t) == 3
        @test_throws ArgumentError intrinsic_dimension(TawnLike{Float64})
        rebuilt = constraint(t, zeros(3))
        @test rebuilt.d == 2
        @test length(rebuilt.dep) == 1
        @test length(rebuilt.weights) == 2
        @test all(w -> sum(w) ≈ 1.0, rebuilt.weights)
    end

    @testset "global coupled geometry" begin
        function ordered_pair_transform()
            base = TV.as((lower=TV.asℝ, gap=TV.asℝ₊))
            forward = p -> (; lower=p.lower, upper=p.lower + p.gap)
            backward = p -> (; lower=p.lower, gap=p.upper - p.lower)
            return joint_transform(base, forward, backward)
        end

        @paramorph T struct OrderedPair{T<:Real}
            lower::T
            upper::T
            @geometry ((lower, upper) ~ ordered_pair_transform())
        end

        pair = OrderedPair(0.0, 2.0)
        @test Paramorph.parameter_fields(typeof(pair)) == (:lower, :upper)
        @test Paramorph.auxiliary_fields(typeof(pair)) == ()
        @test intrinsic_dimension(OrderedPair{Float64}) == 2
        @test_throws DomainError OrderedPair(2.0, 1.0)

        zero_pair = constraint(OrderedPair{Float64}, zeros(2))
        @test zero_pair.lower ≈ 0.0
        @test zero_pair.upper ≈ 1.0
    end

    @testset "constructor validation" begin
        @paramorph T struct Probability{T<:Real}
            p::T ~ bounded_interval(zero(T), one(T))
        end

        @test Probability(0.0).p == 0.0
        @test Probability(1.0).p == 1.0
        @test_throws DomainError Probability(-0.1)
        @test_throws DomainError Probability(1.1)
    end
end

Aqua.test_all(Paramorph)
