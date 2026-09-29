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

# Numeric ownership can be delegated through more than one structural layer.
@paramorph struct NestedOwnedMiddle{C<:NestedOwnedPositive}
    child::C ~ nested()
end

@paramorph struct NestedOwnedOuter{C<:NestedOwnedMiddle}
    child::C ~ nested()
end

# A node may own fit parameters itself and also contain nested fit parameters.
# All of them must follow the coordinate element type during reconstruction.
@paramorph T struct HybridNestedMiddle{T<:Real,C<:NestedOwnedPositive}
    own::T ~ NestedTV.asℝ
    child::C ~ nested()
end

@paramorph struct HybridNestedOuter{C<:HybridNestedMiddle}
    child::C ~ nested()
end

struct OpaqueNestedChild end

# Structural wrappers remain ordinary Julia containers even when a concrete
# child does not opt into Paramorph. Geometry is then unavailable rather than
# construction itself becoming invalid.
@paramorph struct OptionalNestedWrapper{C}
    child::C ~ nested()
end

function _minimum_allocated(f; samples=5)
    f()
    best = typemax(Int)
    for _ in 1:samples
        best = min(best, @allocated f())
    end
    return best
end

function _repeat_constraint(prototype, coordinates, count; context=NamedTuple())
    result = prototype
    for _ in 1:count
        result = constraint(prototype, coordinates; context)
    end
    return result
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

@testset "single nested wrappers stay allocation-light" begin
    child = NestedOwnedScalar(0.2)
    parent = NestedOwnedWrapper{3,typeof(child)}(child)
    coordinates = [0.0]
    repetitions = 64

    child_call = () -> _repeat_constraint(
        child, coordinates, repetitions; context=(; dimension=3),
    )
    parent_call = () -> _repeat_constraint(parent, coordinates, repetitions)

    @test parent_call().child.θ ≈ child_call().θ

    child_bytes = _minimum_allocated(child_call)
    parent_bytes = _minimum_allocated(parent_call)
    @test parent_bytes <= child_bytes + 1024
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

@testset "nested numeric ownership propagates through multiple levels" begin
    leaf = NestedOwnedPositive(2.0)
    middle = NestedOwnedMiddle{typeof(leaf)}(leaf)
    outer = NestedOwnedOuter{typeof(middle)}(middle)

    @test intrinsic_dimension(outer) == 1

    rebuilt = constraint(outer, Float32[0])
    @test rebuilt isa NestedOwnedOuter{
        NestedOwnedMiddle{NestedOwnedPositive{Float32}}
    }
    @test rebuilt.child.child.x ≈ 1.0f0

    typed = constraint(typeof(outer), Float32[0])
    @test typeof(typed) === typeof(rebuilt)
end

@testset "owned and nested parameters share one numeric type" begin
    child = NestedOwnedPositive(2.0)
    middle = HybridNestedMiddle{Float64,typeof(child)}(0.25, child)
    outer = HybridNestedOuter{typeof(middle)}(middle)

    @test intrinsic_dimension(middle) == 2
    @test intrinsic_dimension(outer) == 2

    rebuilt_middle = constraint(middle, Float32[0, 0])
    @test rebuilt_middle isa HybridNestedMiddle{
        Float32,NestedOwnedPositive{Float32}
    }
    @test rebuilt_middle.own ≈ 0.0f0
    @test rebuilt_middle.child.x ≈ 1.0f0

    rebuilt_outer = constraint(outer, Float32[0, 0])
    @test rebuilt_outer isa HybridNestedOuter{
        HybridNestedMiddle{Float32,NestedOwnedPositive{Float32}}
    }
    @test rebuilt_outer.child.own ≈ 0.0f0
    @test rebuilt_outer.child.child.x ≈ 1.0f0

    typed = constraint(typeof(outer), Float32[0, 0])
    @test typeof(typed) === typeof(rebuilt_outer)
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

@testset "public parameter geometry interface" begin
    child = NestedOwnedPositive(2.0)

    @test has_parameter_geometry(NestedOwnedPositive)
    @test has_parameter_geometry(child)
    @test !has_parameter_geometry(OpaqueNestedChild)
    @test !has_parameter_geometry(OpaqueNestedChild())

    @test parameter_values(child) == (; x=2.0)

    prototype = parameter_prototype(NestedOwnedPositive; numeric_type=Float32)
    @test prototype isa NestedOwnedPositive{Float32}
    @test prototype.x ≈ 1.0f0

    contextual = parameter_prototype(
        NestedOwnedScalar;
        numeric_type=Float32,
        context=(; dimension=3),
    )
    @test contextual isa NestedOwnedScalar{Float32}
    @test contextual.θ ≈ 0.5f0

    @test_throws ArgumentError parameter_prototype(OpaqueNestedChild)
end

# Regression coverage for package-integration aliases that arrive as a bare
# UnionAll rather than a concrete numeric specialization.
const NestedOwnedPositiveFamily = NestedOwnedPositive{T} where {T<:Real}

@paramorph T struct AuxiliarySizedNested{T<:Real}
    d::Int
    x::Vector{T} ~ NestedTV.as(Vector, NestedTV.asℝ₊, d)
end

@paramorph struct AuxiliarySizedWrapper{C<:AuxiliarySizedNested}
    child::C ~ nested()
end

@testset "public integration preserves UnionAll and nested auxiliary geometry" begin
    prototype = parameter_prototype(NestedOwnedPositiveFamily; numeric_type=Float32)
    @test prototype isa NestedOwnedPositive{Float32}

    child = AuxiliarySizedNested{Float64}(2, [1.0, 2.0])
    wrapper = AuxiliarySizedWrapper{typeof(child)}(child)
    coordinates = unconstrain(wrapper)

    rebuilt = constraint(wrapper, coordinates)
    @test rebuilt.child.d == 2
    @test rebuilt.child.x ≈ child.x

    rebuilt32 = constraint(wrapper, Float32.(coordinates))
    @test rebuilt32 isa AuxiliarySizedWrapper{AuxiliarySizedNested{Float32}}
    @test rebuilt32.child.d == 2
    @test length(rebuilt32.child.x) == 2
end
