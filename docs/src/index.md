# Paramorph.jl

```@meta
CurrentModule = Paramorph
```

Paramorph declares how the parameter fields of an immutable Julia struct map to
flat unconstrained coordinates.

The 0.1 DSL keeps one central rule:

```julia
field::StorageType ~ geometry
```

`::` describes storage. `~` describes parameter geometry. Fields without `~`
are auxiliary.

See the [tutorial](tutorial.md) for field-dependent geometry, nested structs,
wrappers whose numeric type is carried by nested fields, and coupled
parameterizations.

## Core interface

A model that owns numeric parameter storage declares that numeric type explicitly:

```julia
@paramorph T struct Model{T<:Real}
    parameter::T ~ transform
end

intrinsic_dimension(model)
θ = unconstrain(model)
model′ = constraint(model, θ)
model′, logjac = constraint_with_logjac(model, θ)
```

A structural wrapper may instead carry no numeric type parameter of its own when
its parameter fields are direct nested Paramorph objects stored through the
wrapper's type parameters:

```julia
@paramorph struct Wrapper{C}
    child::C ~ nested()
end
```

In that form the reconstructed wrapper type follows the reconstructed child type.
For example, constraining with `Float32` coordinates may rebuild
`Wrapper{Model{Float64}}` as `Wrapper{Model{Float32}}`. If a concrete nested
child does not declare Paramorph geometry, the wrapper remains an ordinary
constructible Julia object but does not itself expose Paramorph geometry.

Numeric rebinding follows direct `nested` fields recursively. If a node has both
an explicit numeric parameter of its own and a nested Paramorph child stored
through another type parameter, both are rebound to the coordinate element type.
The same rule composes through further nested layers, so one reconstruction uses
one numeric type for all fit-parameter storage reachable through these fields.

`constraint(Type, θ)` is also available when the parameter geometry is fully
determined by the type. If a transformation expression depends on runtime
fields, use a prototype object instead.

`nested(; kwargs...)` makes a nested `@paramorph` field part of the same
parameterization. Its keyword arguments form the local `context` visible to the
child declaration. `recursive([count]; kwargs...)` provides recursive composition
for declarations that have an explicit numeric parameter; numeric-less structural
wrappers in 0.1 use direct `nested` fields.

## Documented building blocks

```@docs
Paramorph.var"@paramorph"
closed_lower
open_lower
nonnegative
bounded_interval
correlation_matrix
closed_correlation_matrix
positive_definite_matrix
variogram_matrix
compact_variogram_matrix
repeat_transform
joint_transform
polytope
```

`positive_vector_with_sum_below` and `asymmetric_mixed` are also exported
specialized transforms; their motivation and use are illustrated by the model
packages that need them.
