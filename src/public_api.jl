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
