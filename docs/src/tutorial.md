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

Nested parameterization is explicit through `nested`. Keyword arguments passed
to `nested` form the local `context` visible in the child declaration. `context`
is a `NamedTuple` supplied by Paramorph; it is not a separate wrapper function.

A parent that owns the same numeric type itself can write:

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

### Numeric type owned by a nested child

A structural wrapper does not need an independent numeric type parameter when
all of its parameter fields are direct nested Paramorph objects. Omit the numeric
argument to `@paramorph` and store each nested field through one of the wrapper's
type parameters:

```@example tutorial
@paramorph T struct DimensionBoundChild{T<:Real}
    θ::T ~ closed_lower(-inv(T(get(context, :dimension, 2) - 1)))
end

@paramorph struct StructuralParent{D,C<:DimensionBoundChild}
    child::C ~ nested(dimension=D)
end

child = DimensionBoundChild(0.2)
parent = StructuralParent{3,typeof(child)}(child)

intrinsic_dimension(parent)
rebuilt = constraint(parent, Float32[0])
typeof(rebuilt)
```

Here `D` is structural information owned by the parent and is forwarded with
`nested(dimension=D)`. The child owns `T` and therefore performs numeric
conversions such as `T(D - 1)` inside its own geometry. Reconstructing from
`Float32` coordinates changes the child from `DimensionBoundChild{Float64}` to
`DimensionBoundChild{Float32}`; the parent concrete type changes accordingly.

The same rule applies to type-based reconstruction when the child types determine
the geometry:

```@example tutorial
constraint(typeof(parent), Float32[0])
```

A structural wrapper's Paramorph capability follows its nested children. This
allows a domain wrapper to accept an opaque extension type without making that
extension implement Paramorph: the wrapper remains constructible, while
`intrinsic_dimension` and coordinate operations are unavailable for that
concrete instance.

Numeric-less `@paramorph struct ...` declarations are intentionally narrow in
0.1: they support field-level `nested(...)` parameters stored through struct type
parameters. Ordinary transformed fields, global `@geometry`, and `recursive`
composition continue to use the explicit `@paramorph T struct ...` form.

For homogeneous vectors of Paramorph objects use `recursive` with an explicit
numeric parameter:

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
