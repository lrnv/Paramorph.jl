# Paramorph.jl

Paramorph attaches an unconstrained parameter geometry to ordinary immutable
Julia structs.

Version **0.0.3** intentionally introduces a breaking DSL. Storage and geometry
are separate concepts:

```julia
using Paramorph
const TV = Paramorph.TransformVariables

@paramorph T struct GaussianParameters{D,T<:Real}
    μ::Vector{T} ~ TV.as(Vector, TV.asℝ, D)
    σ::T ~ TV.asℝ₊
    label::String = "model"
end
```

`::` is ordinary Julia and describes the stored field type. `~` belongs to
Paramorph and describes how that field is represented in unconstrained
coordinates. A field without `~` is auxiliary and is excluded from the
parameter vector.

```julia
p = GaussianParameters{2}([1.0, -1.0], 2.0, "example")
θ = unconstrain(p)
q = constraint(p, θ)
```

## Geometry may depend on the struct

The right-hand side of `~` may refer to type parameters, `context`, and fields
of the struct:

```julia
@paramorph T struct SimplexModel{T<:Real}
    n::Int
    weights::Vector{T} ~ TV.UnitSimplex(n)
end
```

Because the geometry depends on the runtime field `n`, reconstruction needs a
prototype:

```julia
m = SimplexModel(3, [0.2, 0.3, 0.5])
constraint(m, zeros(2))
```

`constraint(SimplexModel{Float64}, zeros(2))` is deliberately rejected: the
type alone does not contain `n`.

When the geometry depends only on type parameters, type-based reconstruction is
available.

## Nested parameterized structs

Nesting is explicit. No `~` means auxiliary, even when the field type was itself
created with `@paramorph`.

```julia
@paramorph T struct Child{T<:Real}
    θ::T ~ closed_lower(get(context, :lower, zero(T)))
end

@paramorph T struct Parent{D,T<:Real}
    child::Child{T} ~ nested(lower=-inv(T(D - 1)))
end
```

`nested(...)` forwards its keyword arguments as the child's local `context`.
Use `recursive(n; ...)` for vectors or other recursive containers of Paramorph
objects.

## Coupled geometries

Independent field transformations are the common case. When several stored
fields form one genuinely coupled parameter geometry, declare it inside the
struct:

```julia
function ordered_pair_transform()
    base = TV.as((lower=TV.asℝ, gap=TV.asℝ₊))
    forward = p -> (; lower=p.lower, upper=p.lower + p.gap)
    backward = p -> (; lower=p.lower, gap=p.upper - p.lower)
    joint_transform(base, forward, backward)
end

@paramorph T struct OrderedPair{T<:Real}
    lower::T
    upper::T
    @geometry ((lower, upper) ~ ordered_pair_transform())
end
```

A struct uses either field-level `~` declarations or one `@geometry` block, not
both. A global geometry must transform to and from a named tuple whose keys are
the listed fields.

## Coordinate API

The core operations are:

```julia
intrinsic_dimension(x)
unconstrain(x)
constraint(x, θ)
constraint(Type, θ)              # when the geometry is type-determined
constraint_with_logjac(x, θ)
```

Direct construction is validated against the same declared geometry.
`constraint` may rebind the explicit numeric type parameter to the element type
of the coordinate vector, which keeps ForwardDiff-style coordinate types
possible.

## Why the tilde?

The 0.0.1/0.0.2 DSL overloaded `::`: Paramorph had to inspect an annotation,
decide whether it was a Julia storage type or a transformation, and infer the
storage type produced by known transformations. That made custom transforms and
value-dependent geometries unnecessarily special.

In 0.0.3 there is no transform-to-storage inference. This is valid without any
Paramorph-specific knowledge of `my_transform`:

```julia
@paramorph T struct CustomModel{T<:Real}
    θ::Vector{T} ~ my_transform()
end
```

The storage contract is written explicitly as `Vector{T}`; the transformation
only owns geometry.
