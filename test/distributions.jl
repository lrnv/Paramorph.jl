using Distributions

@testset "Distributions extension" begin
    independent_examples = (
        Normal(1.0, 2.0), LogNormal(1.0, 2.0), LogitNormal(1.0, 2.0),
        Cauchy(1.0, 2.0), Laplace(1.0, 2.0), Logistic(1.0, 2.0),
        Gumbel(1.0, 2.0), Levy(1.0, 2.0), Gamma(2.0, 3.0),
        Beta(2.0, 3.0), BetaPrime(2.0, 3.0), Frechet(2.0, 3.0),
        InverseGamma(2.0, 3.0), InverseGaussian(2.0, 3.0),
        Kumaraswamy(2.0, 3.0), LogLogistic(2.0, 3.0), Pareto(2.0, 3.0),
        Weibull(2.0, 3.0), FDist(2.0, 3.0), Exponential(2.0), Rayleigh(2.0),
        Chi(2.0), Chisq(2.0), TDist(2.0),
        GeneralizedExtremeValue(1.0, 2.0, 0.3),
        GeneralizedPareto(1.0, 2.0, 0.3), SkewNormal(1.0, 2.0, 0.3),
        NoncentralBeta(2.0, 3.0, 0.5), NoncentralChisq(2.0, 0.5),
        NoncentralF(2.0, 3.0, 0.5), NoncentralT(2.0, 0.5),
        Rician(1.0, 2.0), VonMises(1.0, 2.0), Bernoulli(0.3),
        BernoulliLogit(0.3), Geometric(0.3), NegativeBinomial(2.0, 0.3),
        Poisson(2.0), Skellam(2.0, 3.0), Dirac(1.0),
    )
    for distribution in independent_examples
        coordinates = unconstrain(distribution)
        rebuilt = constraint(distribution, coordinates)
        @test unconstrain(rebuilt) ≈ coordinates
    end

    normal = Normal(2.0, 3.0)
    @test intrinsic_dimension(normal) == 2
    @test all(isapprox.(params(constraint(normal, unconstrain(normal))), params(normal)))
    @test constraint(normal, zeros(2)) == Normal(0.0, 1.0)

    uniform = Uniform(-2.0, 4.0)
    @test constraint(uniform, unconstrain(uniform)) == uniform
    @test intrinsic_dimension(uniform) == 2

    triangular = TriangularDist(-2.0, 4.0, 1.0)
    @test constraint(triangular, unconstrain(triangular)) == triangular
    @test intrinsic_dimension(triangular) == 3

    binomial = Binomial(12, 0.3)
    rebuilt_binomial = constraint(binomial, unconstrain(binomial))
    @test all(isapprox.(params(rebuilt_binomial), params(binomial)))
    @test params(rebuilt_binomial)[1] == 12
    @test intrinsic_dimension(binomial) == 1

    categorical = Categorical([0.2, 0.3, 0.5])
    @test probs(constraint(categorical, unconstrain(categorical))) ≈ probs(categorical)
    @test intrinsic_dimension(categorical) == 2

    @test ForwardDiff.gradient(zeros(2)) do x
        d = constraint(normal, x)
        params(d)[1]^2 + params(d)[2]
    end ≈ [0.0, 1.0]
end
