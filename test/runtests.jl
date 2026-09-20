using Aqua
using Paramorph
using Test
using TransformVariables
using LinearAlgebra
using ForwardDiff

include("copulas_cases.jl")

@testset "Aqua" begin
    Aqua.test_all(Paramorph)
end

@testset "@paramorph definition stress tests" begin
    abstract type AbstractParameters end

    @paramorph struct ScalarOnly
        value::asℝ₊
    end

    @paramorph struct Subtyped{D} <: AbstractParameters
        values::as(Vector, D)
    end

    @paramorph struct MultipleParameters{Rows, Cols}
        matrix::as(Matrix, Rows, Cols)
    end

    @paramorph struct BoundedParameter{Tag<:AbstractString}
        value::asℝ
    end

    @paramorph struct ParametersWithContext{D}
        values::as(Vector, D)
        label::String
        weight::T
    end

    @paramorph struct NestedWithContext{D}
        parameters::ParametersWithContext{D, T}
        scale::asℝ₊
        identifier::Int
    end

    @paramorph struct ParametersWithDefaults{D}
        values::as(Vector, D)
        weight::T = one(T)
        label::String = "default"
    end

    @paramorph struct NestedWithDefaults{D}
        parameters::ParametersWithDefaults{D, T}
        scale::asℝ₊
        identifier::Int = 42
    end

    @paramorph R struct ExplicitNumeric{D,R<:Real}
        values::as(Vector, D)
        label::String = "explicit"
    end

    scalar = ScalarOnly(2.0)
    @test scalar isa ScalarOnly{Float64}
    @test fieldtype(ScalarOnly{Float32}, :value) == Float32
    @test_throws DomainError ScalarOnly(-1.0)

    subtyped = Subtyped{3}([1.0, 2.0, 3.0])
    @test subtyped isa AbstractParameters
    @test subtyped isa Subtyped{3, Float64}
    @test_throws DomainError Subtyped{3}([1.0, 2.0])

    matrix = MultipleParameters{2, 3}(ones(Float32, 2, 3))
    @test matrix isa MultipleParameters{2, 3, Float32}
    @test fieldtype(typeof(matrix), :matrix) == Matrix{Float32}

    bounded = BoundedParameter{String}(1.0)
    @test bounded isa BoundedParameter{String, Float64}

    prototype = ParametersWithContext{2}([1.0, 2.0], "kept", 4.0)
    @test dimension_intrinsique(typeof(prototype)) == 2
    @test unconstrain(prototype) ≈ [1.0, 2.0]
    updated = constraint(prototype, [3.0, 5.0])
    @test updated.values == [3.0, 5.0]
    @test updated.label == "kept"
    @test updated.weight == 4.0
    @test_throws ArgumentError constraint(typeof(prototype), [3.0, 5.0])

    nested = NestedWithContext{2}(prototype, 2.0, 17)
    nested_coordinates = unconstrain(nested)
    @test length(nested_coordinates) == 3
    nested_updated, nested_logjac = constraint_with_logjac(
        nested,
        nested_coordinates .+ 0.1,
    )
    @test nested_updated.parameters.label == "kept"
    @test nested_updated.parameters.weight == 4.0
    @test nested_updated.identifier == 17
    @test nested_updated.parameters.values ≈ [1.1, 2.1]
    @test nested_updated.scale ≈ 2exp(0.1)
    @test isfinite(nested_logjac)

    defaulted = constraint(ParametersWithDefaults{2, Float64}, [3.0, 5.0])
    @test defaulted.values == [3.0, 5.0]
    @test defaulted.weight == 1.0
    @test defaulted.label == "default"
    @test unconstrain(defaulted) == [3.0, 5.0]

    customized = ParametersWithDefaults{2}([1.0, 2.0], 7.0, "custom")
    customized_update = constraint(customized, [4.0, 6.0])
    @test customized_update.weight == 7.0
    @test customized_update.label == "custom"

    nested_defaulted = constraint(NestedWithDefaults{2, Float64}, [1.0, 2.0, 0.0])
    @test nested_defaulted.parameters.weight == 1.0
    @test nested_defaulted.parameters.label == "default"
    @test nested_defaulted.identifier == 42
    @test nested_defaulted.scale == 1.0

    @test constraint(ScalarOnly{Float32}, zeros(Float64, 1)) isa ScalarOnly{Float64}
    @test_throws AssertionError constraint(ScalarOnly{Float64}, zeros(Float64, 2))

    explicit = ExplicitNumeric{2}([1.0, 2.0], "kept")
    @test explicit isa ExplicitNumeric{2,Float64}
    @test constraint(ExplicitNumeric{2,Float32}, zeros(Float64, 2)) isa
        ExplicitNumeric{2,Float64}
    gradient = ForwardDiff.gradient(zeros(2)) do x
        transformed = constraint(explicit, x)
        @test transformed isa ExplicitNumeric{2,<:ForwardDiff.Dual}
        @test transformed.label == "kept"
        return sum(abs2, transformed.values)
    end
    @test gradient == zeros(2)

    mutable_definition = :(@paramorph mutable struct MutableParameters
        value::asℝ
    end)
    untyped_definition = :(@paramorph struct UntypedParameters
        value
    end)
    manual_constructor = :(@paramorph struct ManualConstructor
        value::asℝ
        ManualConstructor(value) = new(value)
    end)
    unknown_transform = :(@paramorph struct UnknownTransform
        value::not_a_transform(1)
    end)
    transformed_default = :(@paramorph struct TransformedDefault
        value::asℝ₊ = 1.0
    end)

    @test_throws ErrorException macroexpand(@__MODULE__, mutable_definition)
    @test_throws ErrorException macroexpand(@__MODULE__, untyped_definition)
    @test_throws ErrorException macroexpand(@__MODULE__, manual_constructor)
    @test_throws ErrorException macroexpand(@__MODULE__, unknown_transform)
    @test_throws ErrorException macroexpand(@__MODULE__, transformed_default)
end

@testset "Main tests" begin
    # N determines the simplex and matrix sizes.
    @paramorph struct DynamicStruct{N}
        a::asℝ₊
        b::UnitSimplex(N)
        C::corr_cholesky_factor(N)
    end

    # Nested constrained structure.
    @paramorph struct ParentStruct{N}
        sub::DynamicStruct{N, T}
        d::asℝ₋
    end

    # D is a phantom parameter: storage remains an ordinary Vector.
    @paramorph struct SizedVector{D}
        values::as(Vector, D)
    end

    # Test with N = 3.
    N_size = 3
    TargetType = ParentStruct{N_size, Float64}

    # a (1) + simplex(3) (2) + correlation factor (3) + d (1) = 7.
    p = dimension_intrinsique(TargetType)
    @test p==7

    # Generate an unconstrained vector in R^7.
    x_input = randn(p)

    # Transform it and compute the log-Jacobian.
    objet, logjac = constraint_with_logjac(TargetType, x_input)

    @test objet.sub.a > 0                  # true
    @test length(objet.sub.b) == N_size    # true (3)
    @test sum(objet.sub.b) ≈ 1.0
    @test size(objet.sub.C) == (3,3)                # (3, 3)
    @test diag(objet.sub.C' * objet.sub.C) ≈ ones(N_size)
    @test objet.d < 0                      # true

    # Round trip.
    x_back = unconstrain(objet)
    @test x_input ≈ x_back

    @test SizedVector{3, Float64}([1.0, 2.0, 3.0]).values == [1.0, 2.0, 3.0]
    @test SizedVector{3}([1.0, 2.0, 3.0]) isa SizedVector{3, Float64}
    @test fieldtype(SizedVector{3, Float64}, :values) == Vector{Float64}
    @test_throws DomainError SizedVector{3, Float64}([1.0, 2.0])
    @test_throws DomainError DynamicStruct{3, Float64}(
        -1.0,
        [0.2, 0.3, 0.5],
        objet.sub.C,
    )
    @test_throws DomainError DynamicStruct{3, Float64}(
        1.0,
        [0.2, 0.3, 0.9],
        objet.sub.C,
    )

    @test transformed_type(asℝ₊, Float32) == Float32
    @test transformed_type(UnitSimplex(3), Float32) == Vector{Float32}
    @test transformed_type(as(Vector, 3), Float64) == Vector{Float64}
    @test transformed_type(as(Matrix, 2, 3), Float64) == Matrix{Float64}
    @test transformed_type(corr_cholesky_factor(3), Float64) ==
        UpperTriangular{Float64, Matrix{Float64}}
    @test fieldtype(DynamicStruct{3, Float32}, :a) == Float32
    @test fieldtype(DynamicStruct{3, Float32}, :b) == Vector{Float32}
    @test fieldtype(DynamicStruct{3, Float32}, :C) ==
        UpperTriangular{Float32, Matrix{Float32}}
end
