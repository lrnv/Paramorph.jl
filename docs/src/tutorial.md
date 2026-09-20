# Tutorial: from unconstrained vectors to constrained structures

The central idea is to keep two representations of the same parameters:

- the **unconstrained** representation is a vector in ``\mathbb R^d``, which is
  convenient for optimizers and MCMC algorithms;
- the **constrained** representation is a readable Julia object whose fields
  automatically satisfy their constraints.

TransformVariables.jl provides the mathematical transformations. Paramorph
adds a macro that associates them with the fields of a structure.

## 1. TransformVariables.jl on its own

A transformation consumes one or more unconstrained real numbers and produces
a constrained value:

```@example tutorial
using TransformVariables

t = asℝ₊
dimension(t)
transform(t, 0.0)
inverse(t, 1.0)
```

Here `asℝ₊` is the exponential transformation, so `0.0` becomes `1.0`. The
four essential operations are:

- `dimension(t)`: the number of unconstrained coordinates consumed by `t`;
- `transform(t, x)`: map unconstrained coordinates to a constrained value;
- `inverse(t, y)`: map a constrained value back to unconstrained coordinates;
- `transform_and_logjac(t, x)`: transform and compute the log absolute
  Jacobian determinant, which is useful in probabilistic calculations.

Transformations can be composed:

```@example tutorial
t = as((
    position = asℝ,
    scale = asℝ₊,
    weights = UnitSimplex(3),
))

dimension(t) # 1 + 1 + (3 - 1)
y = transform(t, zeros(dimension(t)))
inverse(t, y)
```

A simplex of length `N` has only `N - 1` degrees of freedom because its
components are positive and sum to one.

## 2. A first constrained structure

With Paramorph, write only the transformation after `::`. Paramorph appends a
numeric type parameter named `T` to the parameters in the declaration and
derives every stored field type from its transformation and `T`:

```@example tutorial
using Paramorph

@paramorph struct MixtureParameters{N}
    location::asℝ
    scale::asℝ₊
    weights::UnitSimplex(N)
end

P = MixtureParameters{3, Float64} # the macro generated MixtureParameters{N, T}
intrinsic_dimension(P)
```

The macro creates an ordinary Julia structure together with
`transformation_schema(P)`, the TransformVariables transformation associated
with `P`. The value type parameter `N` can therefore determine the simplex
length dynamically.

```@example tutorial
x = zeros(intrinsic_dimension(P))
p = constraint(P, x)

typeof(p)
p.scale
p.weights
unconstrain(p) ≈ x
```

Here `constraint` means “apply the transformation”. It takes a **type** and an
unconstrained vector. Conversely, `unconstrain` takes an **object** and recovers
its unconstrained vector.

### Direct construction is validated too

The macro replaces Julia's automatic constructors with a single inner
constructor. You may provide constrained values directly, but
TransformVariables validates them before the object is created:

```@example tutorial
MixtureParameters{3, Float64}(0.0, 2.0, [0.2, 0.3, 0.5])

try
    MixtureParameters{3, Float64}(0.0, -2.0, [0.2, 0.3, 0.5])
catch error
    error isa DomainError
end
```

Validation also performs a round trip through the transformation. This rejects
an invalid simplex such as `[0.2, 0.3, 0.9]`, even when an isolated call to
`inverse(UnitSimplex(3), ...)` does not check its final sum.

Reconstruction by `constraint` is different: the transformation itself has
already established the invariant, so Paramorph uses a private trusted inner
constructor and does not perform the inverse/round-trip validation again.

A value type parameter can record a dimension without changing the vector's
storage type:

```@example tutorial
@paramorph struct SizedVector{D}
    values::as(Vector, D)
end

v = SizedVector{3}([1.0, 2.0, 3.0]) # T is inferred as Float64 here
typeof(v.values)

try
    SizedVector{3, Float64}([1.0, 2.0])
catch error
    error isa DomainError
end
```

`D` is a phantom parameter: it belongs to the structure type, while `values`
remains an ordinary `Vector{Float64}`. `D` must be specified because Julia
cannot infer it from `Vector{T}` alone, while the generated trailing `T` can be
inferred during direct construction. The constructor guarantees that `D`
agrees with `length(values)`.

### Auxiliary fields and prototypes

A field annotated with an ordinary Julia type is stored in the object but does
not belong to its unconstrained parameter vector:

```@example tutorial
@paramorph struct LabeledMixture{N}
    parameters::MixtureParameters{N, T}
    label::String
    observation_count::Int
end

prototype = LabeledMixture{3}(
    MixtureParameters{3}(0.0, 2.0, [0.2, 0.3, 0.5]),
    "control",
    120,
)

length(unconstrain(prototype)) == intrinsic_dimension(typeof(prototype))
```

`parameters` is another Paramorph structure, so it remains part of the
parameterization. `label` and `observation_count` are auxiliary fields and are
excluded.

A type alone does not contain values for auxiliary fields unless their
declarations provide defaults. Without defaults, reconstruction must use an
existing object as a prototype:

```@example tutorial
x = unconstrain(prototype)
updated = constraint(prototype, x .+ 0.1)

updated.label == prototype.label
updated.observation_count == prototype.observation_count
updated.parameters != prototype.parameters
```

Auxiliary fields are preserved recursively, including inside nested Paramorph
structures. Calling `constraint(typeof(prototype), x)` is rejected because
there is no source for their values. The type-based form remains available for
structures whose complete nested parameterization contains no auxiliary field.

Defaults use Julia's usual `field::Type = value` syntax:

```@example tutorial
@paramorph struct ConfiguredVector{D}
    values::as(Vector, D)
    weight::T = one(T)
    label::String = "default"
end

configured = constraint(ConfiguredVector{3, Float64}, zeros(3))
(configured.weight, configured.label)
```

Default expressions may refer to user type parameters and to the generated
numeric type parameter `T`. They are evaluated when reconstruction occurs.
They apply only to auxiliary fields: transformed fields and nested Paramorph
fields obtain their values from the unconstrained vector and therefore cannot
declare defaults.

When a prototype is supplied, its auxiliary values take precedence over the
declared defaults. This lets one customize an object once and preserve that
context across subsequent transformations:

```@example tutorial
custom = ConfiguredVector{3}(zeros(3), 2.0, "custom")
updated_custom = constraint(custom, ones(3))
(updated_custom.weight, updated_custom.label)
```

Type-based reconstruction also works recursively when every auxiliary field,
including those in nested Paramorph structures, has a default. If any required
default is missing, use the prototype form instead.

## 3. Nested structures

A field without a second annotation is treated as a type that already has a
`transformation_schema`:

```@example tutorial
@paramorph struct ModelParameters{N}
    mixture::MixtureParameters{N, T}
    negative_offset::asℝ₋
end

M = ModelParameters{3, Float64}
x = zeros(intrinsic_dimension(M))
m = constraint(M, x)

(m.mixture.scale, m.mixture.weights, m.negative_offset)
unconstrain(m) ≈ x
```

The total unconstrained dimension is the recursive sum of all field dimensions.

## 4. Cholesky factors of correlation matrices

In TransformVariables 0.8, the relevant constructors are `UnitSimplex(N)` and
`corr_cholesky_factor(N)`. The latter produces an `UpperTriangular`, not a
correlation matrix directly. If `U` is its result, the correlation matrix is
`U' * U`.

```@example tutorial
using LinearAlgebra

@paramorph struct CorrelationParameters{N}
    U::corr_cholesky_factor(N)
end

C = CorrelationParameters{3, Float64}
c = constraint(C, zeros(intrinsic_dimension(C)))
R = c.U' * c.U
diag(R)
```

Its unconstrained dimension is `N * (N - 1) / 2`. The declared field type must
accept the transformation result: `Matrix{T}` alone would not accept an
`UpperTriangular{T, Matrix{T}}` value.

## 5. The log-Jacobian

A density defined on constrained parameters must be corrected when evaluated
in unconstrained coordinates:

```@example tutorial
x = randn(intrinsic_dimension(M))
m, logjac = constraint_with_logjac(M, x)

isfinite(logjac)
unconstrain(m) ≈ x
```

If `logdensity_constrained(m)` is a log density on the constrained object, the
corresponding unconstrained log density is schematically:

```julia
m, logjac = constraint_with_logjac(M, x)
logdensity_unconstrained = logdensity_constrained(m) + logjac
```

Do not add this correction if the statistical library you call already handles
it.

## 6. Quick reference

| Desired value | Transformation | Unconstrained dimension |
|:--|:--|:--|
| real number | `asℝ` | 1 |
| strictly positive real | `asℝ₊` | 1 |
| strictly negative real | `asℝ₋` | 1 |
| real between zero and one | `as𝕀` | 1 |
| vector of `N` reals | `as(Vector, asℝ, N)` | `N` |
| simplex of length `N` | `UnitSimplex(N)` | `N - 1` |
| `N × N` correlation factor | `corr_cholesky_factor(N)` | `N(N - 1)/2` |

The corresponding storage types are computed by
`transformed_type(transformation, T)`. For example:

```@example tutorial
transformed_type(UnitSimplex(3), Float32)
transformed_type(corr_cholesky_factor(3), Float64)
```

Use `methods(as)` to discover generic constructors and Julia's help mode, for
example `?UnitSimplex`, to read the documentation of each transformation.

## 7. Common errors

- The vector passed to `constraint` must contain exactly
  `intrinsic_dimension(T)` elements.
- A transformation must produce a value compatible with its declared field
  type.
- Without an explicit transformation, Paramorph calls
  `transformation_schema(FieldType)` only when `FieldType` is itself a
  Paramorph structure. Other Julia-typed fields are auxiliary. Type-based
  reconstruction requires defaults for all of them; otherwise use a prototype.
- `asSimplex` and `asCorrelationCholesky` are not TransformVariables 0.8
  constructors. Use `UnitSimplex` and `corr_cholesky_factor` instead.

## 8. Advanced schemas for model packages

The default form appends a numeric parameter named `T`. Existing model types
can instead designate one of their declared parameters explicitly:

```@example tutorial
@paramorph R struct ExistingConvention{D,R<:Real}
    values::as(Vector, D)
    label::String = "default"
end

ExistingConvention{2}([1.0, 2.0], "example")
```

`constraint` adopts the element type of its coordinate vector. In particular,
reconstruction from a `Float64` prototype remains compatible with automatic
differentiation: parameter fields may become `ForwardDiff.Dual`, while
auxiliary fields keep their prototype values.

### Closed domains

TransformVariables maps finite coordinates to the interior of a domain.
Paramorph additionally provides transformations whose inverse accepts selected
boundary points:

```@example tutorial
lower = closed_lower(1.0)
TransformVariables.inverse(lower, 1.0)

interval = bounded_interval(-1.0, 1.0)
TransformVariables.inverse(interval, -1.0)
```

Boundaries correspond to infinite coordinates. Consequently, optimizer starts
should normally remain in the interior even when direct construction accepts a
closed endpoint.

### Correlation matrices

`correlation_matrix(D)` wraps TransformVariables' Cholesky-factor chart but
stores the resulting correlation matrix:

```@example tutorial
@paramorph struct GaussianLike{D}
    correlation::correlation_matrix(D)
end

g = constraint(GaussianLike{3,Float64}, zeros(3))
g.correlation
```

It consumes `D * (D - 1) / 2` coordinates and validates symmetry, positive
definiteness, and a unit diagonal during inversion.

### Positive-definite matrices and variograms

`positive_definite_matrix(D)` stores a general symmetric positive-definite
matrix. Its `D * (D + 1) / 2` coordinates parameterize a lower Cholesky factor
with an exponentiated diagonal:

```@example tutorial
S = TransformVariables.transform(positive_definite_matrix(3), zeros(6))
isposdef(S)
```

`variogram_matrix(D)` uses an SPD Gram matrix of size `D - 1` to produce a
strict Hüsler--Reiss variogram: a symmetric, zero-diagonal, conditionally
negative-definite matrix. It has `D * (D - 1) / 2` coordinates:

```@example tutorial
Γ = TransformVariables.transform(variogram_matrix(4), zeros(6))
diag(Γ)
```

Both transformations store ordinary `Matrix{T}` values and support inversion,
automatic differentiation, and log-Jacobian calculation.

### Positive vectors with a bounded sum

`positive_vector_with_sum_below(limit, D)` represents `D` positive values
whose sum is strictly below `limit`. Internally, an additional simplex entry
stores the unused slack:

```@example tutorial
α = TransformVariables.transform(
    positive_vector_with_sum_below(5.0, 3), zeros(3),
)
sum(α) < 5
```

The limit is an ordinary value and may therefore be computed from a parent
context when defining a conditional schema.

### Asymmetric Mixed parameters

`asymmetric_mixed()` is the smooth two-dimensional chart used by the
asymmetric Mixed extreme-value family. It returns a named tuple `(θ₁, θ₂)`
satisfying

```math
θ₁ ≥ 0,\qquad θ₁+θ₂ ≤ 1,\qquad θ₁+2θ₂ ≤ 1,\qquad θ₁+3θ₂ ≥ 0.
```

Unlike the generic radial `polytope` transformation, this family-specific map
is smooth throughout the interior of its admissible quadrilateral.

### Repeated and recursive structures

`repeat_transform(t, n)` applies `t` repeatedly and stores the results in a
vector. This expresses products of simplexes such as the canonical Tawn and
asymmetric Galambos geometries:

```@example tutorial
@paramorph T struct TawnLike{D,T<:Real}
    dependence::as(Vector, closed_lower(one(T)), 2^D-D-1)
    weights::repeat_transform(UnitSimplex(2^(D-1)), D)
end


intrinsic_dimension(TawnLike{3,Float64})
```

Use `recursive(ContainerType)` when a tuple, vector, or named tuple contains
Paramorph objects. A vector additionally needs its length for reconstruction
from a type alone:

```@example tutorial
@paramorph struct PositiveComponent
    strength::asℝ₊
end

@paramorph T struct ComponentVector{N,T<:Real}
    children::recursive(Vector{PositiveComponent{T}}, N)
end

constraint(ComponentVector{2,Float64}, zeros(2))
```

When reconstruction starts from a prototype, the runtime container shape and
all auxiliary values are preserved recursively.

### Parent-dependent schemas

A nested schema can receive context from its parent. Transformation expressions
may read the local `context` named tuple, and a parent supplies it by extending
`Paramorph.schema_context`:

```julia
@paramorph struct DimensionDependent
    value::closed_lower(get(context, :lower, -1.0))
end

Paramorph.schema_context(::Type{<:Parent{D}}, ::Val{:child}) where {D} =
    (; lower=-inv(D - 1))
```

This is useful when a child constraint depends on a parent dimension without
adding that dimension to the child's stored type.

### Joint constraints and polytopes

Independent field transformations are insufficient for constraints such as
`lower < upper` or parent-child inequalities. `joint_transform` constructs a
bidirectional transformation from a base chart and user-supplied forward and
backward maps. A model package can install the resulting whole-object schema by
extending `Paramorph.schema_override` and identifying its parameter fields with
`Paramorph.parameter_fields_override`.

`polytope(A, b; center)` is the corresponding escape hatch for a bounded
polytope `A * y <= b`. It uses radial projection from a strictly interior
point. The mapping is bijective on the interior and supports inversion and
automatic Jacobian calculation, but is only piecewise smooth where the active
facet changes. Model packages should therefore benchmark it against a native
constrained optimizer for difficult likelihoods.

An override may dispatch on the prototype object rather than only its type.
This supports active-face parameterizations: for example, a Liebscher weight
that is structurally zero can remain excluded while the positive entries form
a smaller simplex. The three-argument
`Paramorph.schema_override(Type, context, values)` hook supplies the same
geometry during validating construction, before a prototype exists.

These override hooks are intended for coupled geometries. Ordinary models
should continue to declare transformations directly on their fields.

## 9. Univariate distributions as prototypes

Loading `Distributions.jl` activates Paramorph's optional extension for
univariate distributions. A distribution object acts as a prototype:

```julia
using Distributions, Paramorph

prototype = Normal(2.0, 3.0)
x = unconstrain(prototype)
candidate = constraint(prototype, x)
```

When every constructor argument is fitted and the schema does not depend on an
instance, the distribution family can be used directly:

```julia
normal = constraint(Normal, zeros(intrinsic_dimension(Normal)))
gamma = constraint(Gamma, zeros(intrinsic_dimension(Gamma)))
```

The coordinate element type determines the numeric parameter type, just as it
does for an `@paramorph` structure.

The extension declares the parameter geometry, extracts values through
`Distributions.params`, and reconstructs a candidate with the distribution's
public constructor. Paramorph does not replace or bypass constructors owned by
`Distributions.jl`.

Prototype-based reconstruction is essential when a distribution contains a
structural parameter. For example, the number of trials in a `Binomial` is
kept from the prototype while only its probability is placed in the optimizer
coordinates:

```julia
prototype = Binomial(20, 0.4)
intrinsic_dimension(prototype) == 1
params(constraint(prototype, [0.0]))[1] == 20
```

The initial extension covers the standard scalar families used as copula
margins, ordered-support families such as `Uniform`, the coupled parameters of
`TriangularDist`, and prototype-dependent `Binomial` and `Categorical`
parameters. Multivariate and matrix-variate distributions are intentionally
outside its current scope.

For a foreign type, the same adapter consists of three internal hooks:

```julia
Paramorph.transformation_schema(prototype, context)
Paramorph.parameter_values(prototype)
Paramorph.reconstruct_struct(prototype, constrained_values)
```

The first returns a TransformVariables transformation, the second returns the
corresponding named constrained representation, and the third rebuilds the
foreign object. Model packages should use an extension for these methods so
that optional dependencies remain optional.
