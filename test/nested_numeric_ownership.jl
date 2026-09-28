using Test
using Paramorph

const NestedTV = Paramorph.TransformVariables

@paramorph T struct NestedOwnedScalar{T<:Real}
    θ::T ~ closed_lower(-inv(T(get(context, :dimension, 2) - 1)))
end

# A structural wrapper may own no numeric type parameter itself. Its numerical
# representation is carried entirely by the nested Paramorph child.
@paramorph struct NestedOwnedWrapper{D,C<:NestedOwnedScalar}
    child::C ~ nested(dimension=D)
end

@paramorph T struct NestedOwnedPositive{T<:Real}
    x::T ~ NestedTV.asℝ₊
end

# Cover more than the single-child case: reconstruction must infer the wrapper's
# concrete field types independently from every reconstructed nested child.
@paramorph struct NestedOwnedPair{L<:NestedOwnedPositive,R<:NestedOwnedPositive}
    left::L ~ nested()
    right::R ~ nested()
end

struct OpaqueNestedChild end

# Structural wrappers remain ordinary Julia containers even when a concrete
# child does not opt into Paramorph. Geometry is then unavailable rather than
# construction itself becoming invalid.
@paramorph struct OptionalNestedWrapper{C}
    child::C ~ nested()
end

@testset "nested fields can own the numeric type" begin
    child = NestedOwnedScalar(0.2)
    parent = NestedOwnedWrapper{3,typeof(child)}(child)

    @test intrinsic_dimension(parent) == 1
    @test intrinsic_dimension(typeof(parent)) == 1
    @test unconstrain(constraint(parent, unconstrain(parent))) ≈ unconstrain(parent)

    from_prototype = constraint(parent, Float32[0])
    @test from_prototype isa NestedOwnedWrapper{3,NestedOwnedScalar{Float32}}
    @test from_prototype.child.θ ≈ Float32(0.5)

    from_type = constraint(typeof(parent), Float32[0])
    @test from_type isa NestedOwnedWrapper{3,NestedOwnedScalar{Float32}}
    @test from_type.child.θ ≈ Float32(0.5)

    with_logjac, logjac = constraint_with_logjac(parent, Float32[0])
    @test with_logjac isa NestedOwnedWrapper{3,NestedOwnedScalar{Float32}}
    @test isfinite(logjac)
end

@testset "multiple nested fields determine the rebuilt wrapper type" begin
    pair = NestedOwnedPair{
        NestedOwnedPositive{Float64},NestedOwnedPositive{Float64}
    }(
        NestedOwnedPositive(1.0),
        NestedOwnedPositive(2.0),
    )

    @test intrinsic_dimension(pair) == 2
    rebuilt = constraint(pair, Float32[0, 0])
    @test rebuilt isa NestedOwnedPair{
        NestedOwnedPositive{Float32},NestedOwnedPositive{Float32}
    }
    @test rebuilt.left.x ≈ 1.0f0
    @test rebuilt.right.x ≈ 1.0f0

    typed = constraint(typeof(pair), Float32[0, 0])
    @test typeof(typed) === typeof(rebuilt)
end

@testset "structural geometry follows nested child capability" begin
    opaque = OptionalNestedWrapper{OpaqueNestedChild}(OpaqueNestedChild())
    @test !Paramorph.is_paramorph_type(typeof(opaque))
    @test_throws ArgumentError intrinsic_dimension(opaque)

    parametric = OptionalNestedWrapper{NestedOwnedPositive{Float64}}(
        NestedOwnedPositive(2.0),
    )
    @test Paramorph.is_paramorph_type(typeof(parametric))
    @test Paramorph.supports_type_geometry(typeof(parametric))
    @test intrinsic_dimension(parametric) == 1
end
