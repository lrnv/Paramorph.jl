using Test
using Paramorph

const CompositionTV = Paramorph.TransformVariables

struct FixedComponent
    label::Symbol
end

@paramorph T struct FreeComponent{T<:Real}
    x::T ~ CompositionTV.asℝ
end

@paramorph T struct HeterogeneousComponents{T<:Real}
    children::Tuple{FreeComponent{T},FixedComponent} ~ recursive()
end

@paramorph T struct RuntimeHeterogeneousComponents{T<:Real,C<:AbstractVector}
    children::C ~ recursive()
end

struct ParameterTree{T,C}
    value::T
    children::C
end

struct FixedTreeLeaf
    label::Symbol
end

tree_children(node::ParameterTree) = node.children
tree_children(::FixedTreeLeaf) = ()
tree_value(node::ParameterTree) = node.value
tree_value(::FixedTreeLeaf) = nothing
function tree_transform(node::ParameterTree, parent, ::Int, ::Type{T}) where {T}
    lower = parent === nothing ? zero(T) : convert(T, parent)
    return closed_lower(lower)
end
tree_transform(::FixedTreeLeaf, parent, ::Int, ::Type) = nothing
function rebuild_tree(prototype::ParameterTree, value, children)
    return ParameterTree(value, Tuple(children))
end
rebuild_tree(prototype::FixedTreeLeaf, value, children) = prototype

module ConstructorRegression
using Paramorph
const TV = Paramorph.TransformVariables

function pair_transform()
    base = TV.as((a=TV.asℝ, b=TV.asℝ))
    return joint_transform(base, identity, identity)
end

@paramorph T struct JointPair{T<:Real}
    a::T
    b::T
    @geometry ((a, b) ~ pair_transform())
end

@paramorph T struct VariableTupleParameter{D,T<:Real}
    x::NTuple{D,T} ~ TV.as(ntuple(_ -> TV.asℝ, D))
end
end

@testset "heterogeneous prototype recursion" begin
    prototype = HeterogeneousComponents{Float64}((
        FreeComponent(2.0),
        FixedComponent(:kept),
    ))
    @test intrinsic_dimension(prototype) == 1
    @test unconstrain(prototype) ≈ [2.0]

    rebuilt = constraint(prototype, Float32[3])
    @test rebuilt isa HeterogeneousComponents{Float32}
    @test rebuilt.children[1] isa FreeComponent{Float32}
    @test rebuilt.children[1].x == 3.0f0
    @test rebuilt.children[2] === prototype.children[2]

    vector_prototype = RuntimeHeterogeneousComponents{Float64,Vector{Any}}(
        Any[FreeComponent(2.0), FixedComponent(:vector)],
    )
    vector_rebuilt = constraint(vector_prototype, Float32[4])
    @test vector_rebuilt.children[1] isa FreeComponent{Float32}
    @test vector_rebuilt.children[1].x == 4.0f0
    @test vector_rebuilt.children[2] === vector_prototype.children[2]
end

@testset "joint prototype reconstruction follows constrained numeric type" begin
    prototype = ConstructorRegression.JointPair(1.0, 2.0)
    rebuilt = constraint(prototype, Float32[3, 4])
    @test rebuilt isa ConstructorRegression.JointPair{Float32}
    @test rebuilt.a === 3.0f0
    @test rebuilt.b === 4.0f0
end

@testset "variable NTuple inferred constructor binds numeric type" begin
    x = ConstructorRegression.VariableTupleParameter{2}((1.0, 2.0))
    @test x isa ConstructorRegression.VariableTupleParameter{2,Float64}
    @test isempty(Test.detect_unbound_args(ConstructorRegression; recursive=true))
end

@testset "masked simplex face" begin
    transform = simplex_face((false, true, true))
    value = [0.0, 0.4, 0.6]
    coordinates = CompositionTV.inverse(transform, value)
    @test CompositionTV.dimension(transform) == 1
    @test CompositionTV.transform(transform, coordinates) ≈ value
    @test_throws DomainError CompositionTV.inverse(transform, [0.1, 0.4, 0.5])
    @test_throws DimensionMismatch CompositionTV.inverse(transform, [0.4, 0.6])

    vertex = simplex_face((false, true, false))
    @test CompositionTV.dimension(vertex) == 0
    @test CompositionTV.transform(vertex, Float64[]) == [0.0, 1.0, 0.0]
end

@testset "parent-dependent recursive tree" begin
    prototype = ParameterTree(1.0, (
        ParameterTree(2.0, (ParameterTree(4.0, ()),)),
        FixedTreeLeaf(:opaque),
    ))
    transform = recursive_tree(
        prototype;
        node_transform=tree_transform,
        node_value=tree_value,
        children=tree_children,
        rebuild=rebuild_tree,
    )

    @test CompositionTV.dimension(transform) == 3
    @test CompositionTV.inverse(transform, prototype) ≈ [0.0, 0.0, log(2)]

    rebuilt = CompositionTV.transform(transform, Float32[log(2), 0, log(2)])
    @test rebuilt.value === 2.0f0
    @test rebuilt.children[1].value === 3.0f0
    @test rebuilt.children[1].children[1].value === 5.0f0
    @test rebuilt.children[2] === prototype.children[2]
    @test CompositionTV.inverse(transform, rebuilt) ≈ Float64[log(2), 0, log(2)]
    @test_throws DimensionMismatch CompositionTV.inverse(
        transform,
        ParameterTree(1.0, ()),
    )

    rebuilt_with_logjac, logjac =
        CompositionTV.transform_and_logjac(transform, Float32[log(2), 0, log(2)])
    @test rebuilt_with_logjac == rebuilt
    @test logjac ≈ Float32(log(2) + 0 + log(2))
end
