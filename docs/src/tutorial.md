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

With Paramorph, write the transformation after the field type as a second `::`
annotation:

```@example tutorial
using Paramorph

@paramorph struct MixtureParameters{T, N}
    location::T::asℝ
    scale::T::asℝ₊
    weights::Vector{T}::UnitSimplex(N)
end

P = MixtureParameters{Float64, 3}
dimension_intrinsique(P)
```

The macro creates an ordinary Julia structure together with
`transformation_schema(P)`, the TransformVariables transformation associated
with `P`. The value type parameter `N` can therefore determine the simplex
length dynamically.

```@example tutorial
x = zeros(dimension_intrinsique(P))
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
MixtureParameters{Float64, 3}(0.0, 2.0, [0.2, 0.3, 0.5])

try
    MixtureParameters{Float64, 3}(0.0, -2.0, [0.2, 0.3, 0.5])
catch error
    error isa DomainError
end
```

Validation also performs a round trip through the transformation. This rejects
an invalid simplex such as `[0.2, 0.3, 0.9]`, even when an isolated call to
`inverse(UnitSimplex(3), ...)` does not check its final sum.

A value type parameter can record a dimension without changing the vector's
storage type:

```@example tutorial
@paramorph struct SizedVector{T, D}
    values::Vector{T}::as(Vector, D)
end

v = SizedVector{Float64, 3}([1.0, 2.0, 3.0])
typeof(v.values)

try
    SizedVector{Float64, 3}([1.0, 2.0])
catch error
    error isa DomainError
end
```

`D` is a phantom parameter: it belongs to the structure type, while `values`
remains an ordinary `Vector{Float64}`. You must specify `D` in the constructed
type because Julia cannot infer it from `Vector{T}` alone. The constructor then
guarantees that it agrees with `length(values)`.

## 3. Nested structures

A field without a second annotation is treated as a type that already has a
`transformation_schema`:

```@example tutorial
@paramorph struct ModelParameters{T, N}
    mixture::MixtureParameters{T, N}
    negative_offset::T::asℝ₋
end

M = ModelParameters{Float64, 3}
x = zeros(dimension_intrinsique(M))
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

@paramorph struct CorrelationParameters{T, N}
    U::UpperTriangular{T, Matrix{T}}::corr_cholesky_factor(N)
end

C = CorrelationParameters{Float64, 3}
c = constraint(C, zeros(dimension_intrinsique(C)))
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
x = randn(dimension_intrinsique(M))
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

Use `methods(as)` to discover generic constructors and Julia's help mode, for
example `?UnitSimplex`, to read the documentation of each transformation.

## 7. Common errors

- The vector passed to `constraint` must contain exactly
  `dimension_intrinsique(T)` elements.
- A transformation must produce a value compatible with its declared field
  type.
- Without an explicit transformation, Paramorph calls
  `transformation_schema(FieldType)`. The scalar default is `asℝ`; a nested
  structure must have been declared with `@paramorph`.
- `asSimplex` and `asCorrelationCholesky` are not TransformVariables 0.8
  constructors. Use `UnitSimplex` and `corr_cholesky_factor` instead.
