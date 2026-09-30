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

### One numeric type across nested fit parameters

The coordinate element type is propagated through direct `nested` fields at every
level. This also applies when an intermediate node owns fit parameters itself:
its explicit numeric parameter and the numeric parameters of nested children are
rebound together.

```@example tutorial
@paramorph T struct HybridNode{T<:Real,C<:DimensionBoundChild}
    own::T ~ TV.asℝ
    child::C ~ nested(dimension=3)
end

@paramorph struct HybridOuter{C<:HybridNode}
    child::C ~ nested()
end

hybrid = HybridNode{Float64,DimensionBoundChild{Float64}}(
    0.25,
    DimensionBoundChild(0.2),
)
outer = HybridOuter{typeof(hybrid)}(hybrid)
rebuilt_outer = constraint(outer, Float32[0, 0])
typeof(rebuilt_outer)
```

Here reconstruction from `Float32` coordinates changes both
`HybridNode{Float64,...}` and its `DimensionBoundChild{Float64}` child to their
`Float32` counterparts. The rule composes through additional structural wrappers,
so nested fit parameters do not retain an older numeric type at deeper levels.
The same invariant applies to type-based reconstruction.

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

Prototype-based `recursive()` fields may also contain heterogeneous children.
Children that expose parameter geometry contribute their coordinates; opaque
children are preserved as fixed, zero-dimensional prototype state.

## Prototype-driven trees

Use `recursive_tree` when the topology comes from a prototype and a child's
geometry depends on its newly reconstructed parent. The API is callback-based:
the application supplies accessors for local values and children, a local
transform factory, and a rebuilding function. Paramorph owns traversal,
flattening, inversion, log-Jacobian accumulation, and coordinate numeric types.

```julia
transform = recursive_tree(
    prototype;
    node_transform=(node, parent, index, T) ->
        closed_lower(parent === nothing ? zero(T) : T(parent)),
    node_value=node -> node.value,
    children=node -> node.children,
    rebuild=(node, value, children) -> Node(value, children),
)
```

The topology must remain fixed, and local transform dimensions must not change
when parent values change. Returning `nothing` from `node_transform` makes a
node fixed while recursion may continue through its children.

## Structural simplex faces

`simplex_face(mask)` parameterizes only the active face selected by `mask` and
reinserts exact zeros elsewhere:

```@example tutorial
face = simplex_face((false, true, true))
weights = [0.0, 0.4, 0.6]
TV.transform(face, TV.inverse(face, weights))
```

## Integrating Paramorph from another package

Packages using Paramorph should rely on the high-level public interface rather
than the macro's generated protocol methods.

`has_parameter_geometry` tests whether a concrete type or object currently has a
usable Paramorph geometry. This matters for structural wrappers: a wrapper whose
nested child is opaque may be a perfectly valid domain object while having no
parameter geometry.

`parameter_values` returns the logical constrained values declared by the
parameterization. These values are the constrained counterpart of
`unconstrain(object)` and may differ from raw storage when a declaration uses a
joint or nested geometry.

`parameter_prototype` creates the neutral object associated with zero
unconstrained coordinates for a type-based geometry. It also owns the numeric
rebinding needed to select a concrete floating-point representation:

```@example tutorial
has_parameter_geometry(DimensionBoundChild)

prototype = parameter_prototype(
    DimensionBoundChild;
    numeric_type=Float32,
    context=(; dimension=3),
)
(typeof(prototype), parameter_values(prototype))
```

A domain package can therefore discover geometry, obtain natural declared values,
and create a fitting prototype without depending on Paramorph's internal
`is_paramorph_type`, `parameter_fields`, `numeric_parameter_index`, or
`rebind_numeric_type` protocol. Errors from a declared but broken geometry are
not converted into absence: `has_parameter_geometry` only answers the capability
question, while the actual geometry operations still propagate their errors.

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

## Whole-object validation

Field geometries describe the values produced from unconstrained coordinates.
Use `@validate` when validity also depends on a relation between reconstructed
fields:

```@example tutorial
@paramorph T struct OrderedScales{T<:Real}
    lower::T ~ TV.asℝ₊
    upper::T ~ TV.asℝ₊
    @validate lower < upper
end

OrderedScales(1.0, 2.0)
```

The predicate runs for direct construction and after type- or prototype-based
reconstruction. It may reference every field and `context`. It must return
`true`, `false`, or `nothing`: `false` produces a `DomainError`, while `nothing`
allows validation functions that signal failure by throwing their own
domain-specific exception.

```@example tutorial
constraint(OrderedScales{Float64}, [0.0, 1.0])
```

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
