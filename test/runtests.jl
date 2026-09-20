using Aqua
using Paramorph
using Test
using TransformVariables
using LinearAlgebra

@testset "Aqua" begin
    Aqua.test_all(Paramorph)
end

@testset "Main tests" begin
    # 1. Structure dynamique (N définit la taille du simplexe et de la matrice)
    @constrained_struct struct DynamicStruct{T, N}
        a::T::asℝ₊
        b::Vector{T}::asSimplex(N)
        C::Matrix{T}::asCorrelationCholesky(N)
    end

    # 2. Structure qui imbrique la structure dynamique
    @constrained_struct struct ParentStruct{T, N}
        sub::DynamicStruct{T, N}   # Reçoit automatiquement N de l'appelant
        d::T::asℝ₋
    end

    # --- TEST AVEC N = 3 ---
    N_size = 3
    TargetType = ParentStruct{Float64, N_size}

    # Calcul de la dimension intrinsèque totale pour N = 3 :
    # a (1) + simplex(3) (2) + corr(3) (3) + d (1) = 7
    p = dimension_intrinsique(TargetType)
    @test p==7

    # On génère un vecteur aléatoire dans R^7
    x_input = randn(p)

    # Mapping complet avec récupération du log-Jacobien !
    objet, logjac = constraint_with_logjac(TargetType, x_input)

    @test objet.sub.a > 0                  # true
    @test length(objet.sub.b) == N_size    # true (3)
    @test sum(objet.sub.b) ≈ 1.0           # true (Propriété du simplexe)
    @test size(objet.sub.C) == (3,3)                # (3, 3)
    @test all(diag(objet.sub.C) .≈ 1.0)    # true (Propriété d'une matrice de corrélation)
    @test objet.d < 0                      # true

    # --- Aller-Retour ---
    x_back = unconstrain(objet)
    @test x_input ≈ x_back
end
