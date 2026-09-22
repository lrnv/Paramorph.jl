# Paramorph.jl

```@meta
CurrentModule = Paramorph
```

Paramorph declares how the parameter fields of an immutable Julia struct map to
flat unconstrained coordinates.

The 0.0.3 DSL has one central rule:

```julia
field::StorageType ~ geometry
```

`::` describes storage. `~` describes parameter geometry. Fields without `~`
are auxiliary.

See the [tutorial](tutorial.md) for field-dependent geometry, nested structs and
coupled parameterizations.

## Core interface

The core interface consists of the declaration macro and the coordinate
operations described throughout the tutorial:

```julia
@paramorph T struct Model{T<:Real}
    parameter::T ~ transform
end

intrinsic_dimension(model)
θ = unconstrain(model)
model′ = constraint(model, θ)
model′, logjac = constraint_with_logjac(model, θ)
```

`constraint(Type, θ)` is also available when the parameter geometry is fully
determined by the type. If a transformation expression depends on runtime
fields, use a prototype object instead.

`nested(; context...)` makes a nested `@paramorph` field part of the same
parameterization and forwards a local context. `recursive([count]; context...)`
does the same for recursive containers.

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
