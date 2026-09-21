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

## Public API

```@docs
Paramorph.var"@paramorph"
constraint
constraint_with_logjac
unconstrain
intrinsic_dimension
nested
recursive
closed_lower
nonnegative
bounded_interval
correlation_matrix
positive_definite_matrix
variogram_matrix
positive_vector_with_sum_below
asymmetric_mixed
repeat_transform
joint_transform
polytope
```
