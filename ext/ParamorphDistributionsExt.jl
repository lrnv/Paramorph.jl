module ParamorphDistributionsExt

import Distributions
import Paramorph
import TransformVariables

const TV = TransformVariables

_named_schema(names, transforms) = TV.as(NamedTuple{names}(transforms))
_named_values(names, values) = NamedTuple{names}(values)

# Families for which every constructor argument is a continuous parameter.
# Names are declared explicitly because storage field names are not part of the
# public Distributions.jl interface; `params` supplies the constructor order.
const _INDEPENDENT_FAMILIES = (
    (Distributions.Normal, (:μ, :σ), (TV.asℝ, TV.asℝ₊)),
    (Distributions.LogNormal, (:μ, :σ), (TV.asℝ, TV.asℝ₊)),
    (Distributions.LogitNormal, (:μ, :σ), (TV.asℝ, TV.asℝ₊)),
    (Distributions.Cauchy, (:μ, :σ), (TV.asℝ, TV.asℝ₊)),
    (Distributions.Laplace, (:μ, :θ), (TV.asℝ, TV.asℝ₊)),
    (Distributions.Logistic, (:μ, :θ), (TV.asℝ, TV.asℝ₊)),
    (Distributions.Gumbel, (:μ, :θ), (TV.asℝ, TV.asℝ₊)),
    (Distributions.Levy, (:μ, :σ), (TV.asℝ, TV.asℝ₊)),
    (Distributions.Gamma, (:α, :θ), (TV.asℝ₊, TV.asℝ₊)),
    (Distributions.Beta, (:α, :β), (TV.asℝ₊, TV.asℝ₊)),
    (Distributions.BetaPrime, (:α, :β), (TV.asℝ₊, TV.asℝ₊)),
    (Distributions.Frechet, (:α, :θ), (TV.asℝ₊, TV.asℝ₊)),
    (Distributions.InverseGamma, (:α, :θ), (TV.asℝ₊, TV.asℝ₊)),
    (Distributions.InverseGaussian, (:μ, :λ), (TV.asℝ₊, TV.asℝ₊)),
    (Distributions.Kumaraswamy, (:a, :b), (TV.asℝ₊, TV.asℝ₊)),
    (Distributions.LogLogistic, (:α, :β), (TV.asℝ₊, TV.asℝ₊)),
    (Distributions.Pareto, (:α, :θ), (TV.asℝ₊, TV.asℝ₊)),
    (Distributions.Weibull, (:α, :θ), (TV.asℝ₊, TV.asℝ₊)),
    (Distributions.FDist, (:ν1, :ν2), (TV.asℝ₊, TV.asℝ₊)),
    (Distributions.Exponential, (:θ,), (TV.asℝ₊,)),
    (Distributions.Rayleigh, (:σ,), (TV.asℝ₊,)),
    (Distributions.Chi, (:ν,), (TV.asℝ₊,)),
    (Distributions.Chisq, (:ν,), (TV.asℝ₊,)),
    (Distributions.TDist, (:ν,), (TV.asℝ₊,)),
    (Distributions.GeneralizedExtremeValue, (:μ, :σ, :ξ), (TV.asℝ, TV.asℝ₊, TV.asℝ)),
    (Distributions.GeneralizedPareto, (:μ, :σ, :ξ), (TV.asℝ, TV.asℝ₊, TV.asℝ)),
    (Distributions.SkewNormal, (:ξ, :ω, :α), (TV.asℝ, TV.asℝ₊, TV.asℝ)),
    (Distributions.NoncentralBeta, (:α, :β, :λ), (TV.asℝ₊, TV.asℝ₊, Paramorph.nonnegative())),
    (Distributions.NoncentralChisq, (:ν, :λ), (TV.asℝ₊, Paramorph.nonnegative())),
    (Distributions.NoncentralF, (:ν1, :ν2, :λ), (TV.asℝ₊, TV.asℝ₊, Paramorph.nonnegative())),
    (Distributions.NoncentralT, (:ν, :λ), (TV.asℝ₊, TV.asℝ)),
    (Distributions.Rician, (:ν, :σ), (Paramorph.nonnegative(), TV.asℝ₊)),
    (Distributions.VonMises, (:μ, :κ), (TV.asℝ, Paramorph.nonnegative())),
    (Distributions.Bernoulli, (:p,), (Paramorph.bounded_interval(0, 1),)),
    (Distributions.BernoulliLogit, (:logitp,), (TV.asℝ,)),
    (Distributions.Geometric, (:p,), (Paramorph.bounded_interval(0, 1; left_closed=false),)),
    (Distributions.NegativeBinomial, (:r, :p), (TV.asℝ₊, Paramorph.bounded_interval(0, 1; left_closed=false))),
    (Distributions.Poisson, (:λ,), (Paramorph.nonnegative(),)),
    (Distributions.Skellam, (:μ1, :μ2), (Paramorph.nonnegative(), Paramorph.nonnegative())),
)

for (family, names, transforms) in _INDEPENDENT_FAMILIES
    @eval begin
        Paramorph._supports_type_reconstruction(::Type{<:$family}) = true
        Paramorph.transformation_schema(::Type{<:$family}) =
            _named_schema($names, $transforms)
        Paramorph.transformation_schema(::Type{<:$family}, ::NamedTuple) =
            _named_schema($names, $transforms)
        Paramorph.transformation_schema(::$family, ::NamedTuple) =
            _named_schema($names, $transforms)
        Paramorph.parameter_values(d::$family) =
            _named_values($names, Distributions.params(d))
        Paramorph.reconstruct_struct(::Type{<:$family}, values::NamedTuple) =
            $family(Tuple(values)...)
        Paramorph.reconstruct_struct(::$family, values::NamedTuple) =
            $family(Tuple(values)...)
    end
end

Paramorph._supports_type_reconstruction(::Type{<:Distributions.Dirac}) = true
Paramorph.transformation_schema(::Type{<:Distributions.Dirac}) =
    _named_schema((:x,), (TV.asℝ,))
Paramorph.transformation_schema(::Type{<:Distributions.Dirac}, ::NamedTuple) =
    _named_schema((:x,), (TV.asℝ,))
Paramorph.transformation_schema(::Distributions.Dirac, ::NamedTuple) =
    _named_schema((:x,), (TV.asℝ,))
Paramorph.parameter_values(d::Distributions.Dirac) = (; x=d.value)
Paramorph.reconstruct_struct(::Type{<:Distributions.Dirac}, p::NamedTuple) =
    Distributions.Dirac(p.x)
Paramorph.reconstruct_struct(::Distributions.Dirac, p::NamedTuple) =
    Distributions.Dirac(p.x)

function _ordered_schema()
    base = TV.as((first=TV.asℝ, gap=TV.asℝ₊))
    forward = p -> (; a=p.first, b=p.first + p.gap)
    backward = p -> (; first=p.a, gap=p.b - p.a)
    return Paramorph.joint_transform(base, forward, backward)
end

for family in (Distributions.Uniform, Distributions.Arcsine)
    @eval begin
        Paramorph._supports_type_reconstruction(::Type{<:$family}) = true
        Paramorph.transformation_schema(::Type{<:$family}) = _ordered_schema()
        Paramorph.transformation_schema(::Type{<:$family}, ::NamedTuple) = _ordered_schema()
        Paramorph.transformation_schema(::$family, ::NamedTuple) = _ordered_schema()
        Paramorph.parameter_values(d::$family) = _named_values((:a, :b), Distributions.params(d))
        Paramorph.reconstruct_struct(::Type{<:$family}, p::NamedTuple) = $family(p.a, p.b)
        Paramorph.reconstruct_struct(::$family, p::NamedTuple) = $family(p.a, p.b)
    end
end

function _triangular_schema()
    base = TV.as((a=TV.asℝ, gap=TV.asℝ₊, fraction=Paramorph.bounded_interval(0, 1)))
    forward = p -> (; a=p.a, b=p.a + p.gap, c=p.a + p.fraction * p.gap)
    backward = p -> (; a=p.a, gap=p.b - p.a, fraction=(p.c - p.a) / (p.b - p.a))
    return Paramorph.joint_transform(base, forward, backward)
end

Paramorph._supports_type_reconstruction(::Type{<:Distributions.TriangularDist}) = true
Paramorph.transformation_schema(::Type{<:Distributions.TriangularDist}) =
    _triangular_schema()
Paramorph.transformation_schema(
    ::Type{<:Distributions.TriangularDist}, ::NamedTuple,
) = _triangular_schema()
Paramorph.transformation_schema(::Distributions.TriangularDist, ::NamedTuple) =
    _triangular_schema()
Paramorph.parameter_values(d::Distributions.TriangularDist) =
    _named_values((:a, :b, :c), Distributions.params(d))
Paramorph.reconstruct_struct(::Type{<:Distributions.TriangularDist}, p::NamedTuple) =
    Distributions.TriangularDist(p.a, p.b, p.c)
Paramorph.reconstruct_struct(::Distributions.TriangularDist, p::NamedTuple) =
    Distributions.TriangularDist(p.a, p.b, p.c)

# Integer counts and bounds are structural: the prototype supplies them and
# only the continuous coordinates are reconstructed.
Paramorph.transformation_schema(::Distributions.Binomial, ::NamedTuple) =
    _named_schema((:p,), (Paramorph.bounded_interval(0, 1),))
Paramorph.parameter_values(d::Distributions.Binomial) = (; p=Distributions.params(d)[2])
Paramorph.reconstruct_struct(d::Distributions.Binomial, p::NamedTuple) =
    Distributions.Binomial(Distributions.params(d)[1], p.p)

Paramorph.transformation_schema(::Distributions.BetaBinomial, ::NamedTuple) =
    _named_schema((:α, :β), (TV.asℝ₊, TV.asℝ₊))
Paramorph.parameter_values(d::Distributions.BetaBinomial) = begin
    _, α, β = Distributions.params(d)
    (; α, β)
end
Paramorph.reconstruct_struct(d::Distributions.BetaBinomial, p::NamedTuple) =
    Distributions.BetaBinomial(Distributions.params(d)[1], p.α, p.β)

function Paramorph.transformation_schema(d::Distributions.Categorical, ::NamedTuple)
    return _named_schema((:p,), (TV.UnitSimplex(length(Distributions.probs(d))),))
end
Paramorph.parameter_values(d::Distributions.Categorical) = (; p=Distributions.probs(d))
Paramorph.reconstruct_struct(::Distributions.Categorical, p::NamedTuple) =
    Distributions.Categorical(p.p)

end
