export has_parameter_geometry, parameter_values, parameter_prototype

"""
    has_parameter_geometry(T::Type)
    has_parameter_geometry(object)

Return whether `T` or `object` currently exposes a Paramorph parameter geometry.
For structural wrappers this capability may depend on the concrete nested child
types. This function only answers the capability question; errors from an
available but broken geometry still propagate from geometry operations.
"""
has_parameter_geometry(T::Type) = is_paramorph_type(T)
has_parameter_geometry(object) = has_parameter_geometry(typeof(object))

"""
    parameter_values(object; context=NamedTuple())

Return the logical constrained parameter values declared by `object` as a named
tuple. These are the constrained counterpart of `unconstrain(object)` and can
differ from raw storage for nested or joint parameterizations.
"""
function parameter_values end

# A bare UnionAll family can reach the public integration API when a domain
# package stores a parametric family alias rather than a concrete leaf type.
# Dispatching the generic Type method on that UnionAll can miss the generated
# numeric-parameter metadata, so recover it from the unwrapped DataType before
# rebuilding the requested numeric specialization.
function rebind_numeric_type(T::UnionAll, ::Type{N}) where {N}
    U = Base.unwrap_unionall(T)
    index = numeric_parameter_index(U)
    index === nothing && return T
    parameters = collect(U.parameters)
    index <= length(parameters) || return T
    parameters[index] = N
    return Core.apply_type(Base.typename(U).wrapper, parameters...)
end

# The dense prototype fast path is valid only when the complete nested object
# graph can derive its geometry from types alone. A structural parent can itself
# have no auxiliary fields while containing a child whose geometry depends on
# stored auxiliary state, so inspect nested parameter values recursively before
# selecting the type-only schema.
function _value_supports_type_geometry(value)
    if has_parameter_geometry(value)
        T = typeof(value)
        supports_type_geometry(T) || return false
        isempty(auxiliary_fields(T)) || return false
        return all(parameter_fields(T)) do field
            _value_supports_type_geometry(getfield(value, field))
        end
    elseif value isa Tuple || value isa AbstractArray
        return all(_value_supports_type_geometry, value)
    end
    return true
end

function _prototype_supports_type_geometry(target::Type, prototype)
    supports_type_geometry(target) || return false
    isempty(auxiliary_fields(target)) || return false
    return all(parameter_fields(target)) do field
        _value_supports_type_geometry(getfield(prototype, field))
    end
end

# Optimizers overwhelmingly pass dense coordinate vectors. When a prototype's
# geometry is already fully determined by its concrete type and local context,
# rebuilding that schema from every stored field only adds work. In particular,
# purely structural `nested(...)` wrappers can reuse their type geometry while
# still reconstructing from the prototype so auxiliary storage is preserved.
function _prototype_constraint_schema(target::Type, prototype, context::NamedTuple)
    if _prototype_supports_type_geometry(target, prototype)
        return transformation_schema(target, context)
    end
    return _schema_from_values(target, _struct_values(prototype), context)
end

# These dense-vector methods are more specific than the generic AbstractVector
# implementations in Paramorph.jl. The Type variants avoid dispatch ambiguity
# and intentionally preserve the ordinary type-based path unchanged.
function constraint(
    T::Type, coordinates::Vector{<:Real};
    context=NamedTuple(), auxiliary=NamedTuple(),
)
    target = rebind_numeric_type(T, eltype(coordinates))
    schema = _type_schema(target, context, auxiliary)
    length(coordinates) == TransformVariables.dimension(schema) || throw(DimensionMismatch(
        "expected $(TransformVariables.dimension(schema)) coordinates, got $(length(coordinates))",
    ))
    constrained = TransformVariables.transform(schema, coordinates)
    return _reconstruct_declared_from_type(target, constrained, context, auxiliary)
end

function constraint(prototype, coordinates::Vector{<:Real}; context=NamedTuple())
    target = rebind_numeric_type(typeof(prototype), eltype(coordinates))
    schema = _prototype_constraint_schema(target, prototype, context)
    length(coordinates) == TransformVariables.dimension(schema) || throw(DimensionMismatch(
        "expected $(TransformVariables.dimension(schema)) coordinates, got $(length(coordinates))",
    ))
    constrained = TransformVariables.transform(schema, coordinates)
    return _reconstruct_declared_from_prototype(prototype, target, constrained, context)
end

function constraint_with_logjac(
    T::Type, coordinates::Vector{<:Real};
    context=NamedTuple(), auxiliary=NamedTuple(),
)
    target = rebind_numeric_type(T, eltype(coordinates))
    schema = _type_schema(target, context, auxiliary)
    length(coordinates) == TransformVariables.dimension(schema) || throw(DimensionMismatch(
        "expected $(TransformVariables.dimension(schema)) coordinates, got $(length(coordinates))",
    ))
    constrained, logjac = TransformVariables.transform_and_logjac(schema, coordinates)
    return _reconstruct_declared_from_type(target, constrained, context, auxiliary), logjac
end

function constraint_with_logjac(
    prototype, coordinates::Vector{<:Real}; context=NamedTuple(),
)
    target = rebind_numeric_type(typeof(prototype), eltype(coordinates))
    schema = _prototype_constraint_schema(target, prototype, context)
    length(coordinates) == TransformVariables.dimension(schema) || throw(DimensionMismatch(
        "expected $(TransformVariables.dimension(schema)) coordinates, got $(length(coordinates))",
    ))
    constrained, logjac = TransformVariables.transform_and_logjac(schema, coordinates)
    return _reconstruct_declared_from_prototype(prototype, target, constrained, context), logjac
end

"""
    parameter_prototype(T::Type; numeric_type=Float64, context=NamedTuple(), auxiliary=NamedTuple())

Construct the neutral type-based prototype of `T` by using zero unconstrained
coordinates. `numeric_type` selects the concrete real type used for reconstructed
fit parameters. `context` and `auxiliary` have the same meaning as in type-based
`intrinsic_dimension` and `constraint`.

This is the high-level package-integration entry point for creating a fitting
prototype; callers should not depend on Paramorph's internal numeric type
rebinding protocol.
"""
function parameter_prototype(
    T::Type;
    numeric_type::Type{N}=Float64,
    context=NamedTuple(),
    auxiliary=NamedTuple(),
) where {N<:Real}
    target = rebind_numeric_type(T, N)
    has_parameter_geometry(target) || throw(ArgumentError(
        "$T does not declare a Paramorph parameter geometry",
    ))
    n = intrinsic_dimension(target; context, auxiliary)
    return constraint(target, zeros(N, n); context, auxiliary)
end
