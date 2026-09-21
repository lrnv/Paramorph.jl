# Tutorial

```@setup tutorial
using Paramorph
const TV = Paramorph.TransformVariables
```

## Storage and geometry

A Paramorph declaration keeps ordinary Julia storage types intact and places the
constraint after a tilde:

```@example tutorial
@paramorph T struct Scale{T<:Real}
    value::T ~ TV.asℝ₊
    label::String = "default"
end

s = Scale{Float64}(2.0, "custom")
θ = unconstrain(s)
constraint(s, θ)
```

`label` has no `~`, so it is auxiliary and does not consume optimizer
coordinates. Structs with auxiliary fields intentionally do not receive an
inferred unparameterized outer constructor: use the fully parameterized
constructor, as above, or define a domain-specific outer constructor.

Type-based reconstruction is possible when geometry is determined entirely by
the type:

```@example tutorial
constraint(Scale{Float64}, [0.0])
```

The auxiliary default is used because there is no prototype from which to copy
`label`.

## Geometry depending on runtime fields

Transformation expressions may refer directly to fields:

```@example tutorial
@paramorph T struct Weights{T<:Real}
    n::Int
    value::Vector{T} ~ TV.UnitSimplex(n)
end

w = Weights{Float64}(3, [0.2, 0.3, 0.5])
intrinsic_dimension(w)
constraint(w, zeros(2))
```

Here `n` is only stored at runtime, so `Weights{Float64}` cannot by itself
identify a parameter space. Use a prototype object for `constraint` and
`intrinsic_dimension`, or pass the auxiliary field explicitly to type-based
operations.

## Type-dependent geometry

A type parameter may be referenced without requiring a prototype:

```@example tutorial
@paramorph T struct FixedWeights{N,T<:Real}
    value::Vector{T} ~ TV.UnitSimplex(N)
end

intrinsic_dimension(FixedWeights{4,Float64})
```

## Nested parameterizations and context

Nested parameterization is explicit through `nested`. Keyword arguments form a
local `context` visible in the child declaration:

```@example tutorial
@paramorph T struct Child{T<:Real}
    θ::T ~ closed_lower(get(context, :lower, zero(T)))
end

@paramorph T struct Parent{D,T<:Real}
    child::Child{T} ~ nested(lower=-inv(T(D - 1)))
end

p = constraint(Parent{3,Float64}, [0.0])
p.child.θ
```

This keeps parent-dependent constraints next to the field declaration instead
of requiring external `schema_context` methods.

For homogeneous vectors of Paramorph objects use `recursive`:

```@example tutorial
@paramorph T struct Forest{N,T<:Real}
    children::Vector{Child{T}} ~ recursive(N; lower=zero(T))
end

intrinsic_dimension(Forest{2,Float64})
```

## Coupled fields

Some geometries are genuinely joint rather than Cartesian products of field
constraints. Use one `@geometry` declaration inside the struct:

```@example tutorial
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

pair = OrderedPair(0.0, 2.0)
unconstrain(pair)
constraint(OrderedPair{Float64}, zeros(2))
```

The transformation must use a named tuple with the fields listed in
`@geometry`. Field-level `~` declarations and `@geometry` intentionally cannot
be mixed in the same struct.

## Custom transforms need no registration

Because the storage type is explicit, Paramorph does not need a registry that
knows the output type of every transformation:

```julia
@paramorph T struct MyModel{T<:Real}
    parameter::Vector{T} ~ my_custom_transform()
end
```

If `my_custom_transform()` satisfies the TransformVariables transformation
interface and produces a value compatible with `Vector{T}`, no Paramorph method
is required.
