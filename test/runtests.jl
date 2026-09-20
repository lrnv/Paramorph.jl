using Aqua
using Paramorph
using Test
using TransformVariables
using LinearAlgebra

@testset "Aqua" begin
    Aqua.test_all(Paramorph)
end

@testset "Main tests" begin
    # N determines the simplex and matrix sizes.
    @paramorph struct DynamicStruct{T, N}
        a::T::asℝ₊
        b::Vector{T}::UnitSimplex(N)
        C::UpperTriangular{T, Matrix{T}}::corr_cholesky_factor(N)
    end

    # Nested constrained structure.
    @paramorph struct ParentStruct{T, N}
        sub::DynamicStruct{T, N}
        d::T::asℝ₋
    end

    # D is a phantom parameter: storage remains an ordinary Vector.
    @paramorph struct SizedVector{T, D}
        values::Vector{T}::as(Vector, D)
    end

    # Test with N = 3.
    N_size = 3
    TargetType = ParentStruct{Float64, N_size}

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

    @test SizedVector{Float64, 3}([1.0, 2.0, 3.0]).values == [1.0, 2.0, 3.0]
    @test fieldtype(SizedVector{Float64, 3}, :values) == Vector{Float64}
    @test_throws DomainError SizedVector{Float64, 3}([1.0, 2.0])
    @test_throws DomainError DynamicStruct{Float64, 3}(
        -1.0,
        [0.2, 0.3, 0.5],
        objet.sub.C,
    )
    @test_throws DomainError DynamicStruct{Float64, 3}(
        1.0,
        [0.2, 0.3, 0.9],
        objet.sub.C,
    )
end
