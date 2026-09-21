module Paramorph

using LinearAlgebra
import TransformVariables

export @paramorph,
    constraint,
    constraint_with_logjac,
    unconstrain,
    intrinsic_dimension,
    nested,
    recursive,
    closed_lower,
    nonnegative,
    bounded_interval,
    correlation_matrix,
    positive_definite_matrix,
    variogram_matrix,
    positive_vector_with_sum_below,
    asymmetric_mixed,
    repeat_transform,
    joint_transform,
    polytope

include("transforms.jl")

# -----------------------------------------------------------------------------
# Geometry specifications
# -----------------------------------------------------------------------------

"""
    nested(; context...)

Declare a field as a nested Paramorph object. Keyword arguments are forwarded as
that object's local geometry context.
"""
struct NestedSpec{C}
    context::C
end
nested(; kwargs...) = NestedSpec((; kwargs...))

"""
    recursive([count]; context...)

Declare a collection of nested Paramorph objects. If `count` is supplied it is
used for type-based reconstruction. Keyword arguments are forwarded to each
child as local geometry context.
"""
struct RecursiveSpec{C,N}
    context::C
    count::N
end
recursive(; kwargs...) = RecursiveSpec((; kwargs...), nothing)
recursive(count; kwargs...) = RecursiveSpec((; kwargs...), count)

_materialize_geometry(geometry, ::Type, value, ::Bool) = geometry
_materialize_geometry(spec::NestedSpec, ::Type{T}, value, have_value::Bool) where {T} =
    have_value ? recursive_schema(value, spec.context) : recursive_schema(T, spec.context)
_materialize_geometry(spec::RecursiveSpec, ::Type{T}, value, have_value::Bool) where {T} =
    _recursive_collection_schema(spec, T, value, have_value)

function _recursive_collection_schema(spec::RecursiveSpec, ::Type{T}, value, have_value::Bool) where {T}
    if have_value
        children = collect(value)
        schemas = map(children) do child
            recursive_schema(child, spec.context)
        end
        return TransformVariables.as(Tuple(schemas))
    end

    spec.count === nothing && throw(ArgumentError(
        "type-based recursive geometry for $T needs `recursive(count; ...)`",
    ))
    E = eltype(T)
    schemas = ntuple(_ -> recursive_schema(E, spec.context), Int(spec.count))
    return TransformVariables.as(schemas)
end

recursive_schema(object, context::NamedTuple=NamedTuple()) =
    transformation_schema(object, context)
recursive_schema(T::Type, context::NamedTuple=NamedTuple()) =
    transformation_schema(T, context)

function _recursive_constraint(spec::NestedSpec, ::Type{T}, constrained, prototype, have_prototype::Bool) where {T}
    if have_prototype
        return _reconstruct_declared_from_prototype(prototype, typeof(prototype), constrained, spec.context)
    end
    return _reconstruct_declared_from_type(T, constrained, spec.context)
end

function _recursive_constraint(spec::RecursiveSpec, ::Type{T}, constrained, prototype, have_prototype::Bool) where {T}
    E = eltype(T)
    if have_prototype
        return [
            _reconstruct_declared_from_prototype(prototype[i], typeof(prototype[i]), constrained[i], spec.context)
            for i in eachindex(constrained)
        ]
    end
    return [
        _reconstruct_declared_from_type(E, constrained[i], spec.context)
        for i in eachindex(constrained)
    ]
end

_recursive_parameter_value(::NestedSpec, value, context) =
    parameter_values(value; context)
_recursive_parameter_value(spec::RecursiveSpec, value, context) =
    Tuple(parameter_values(child; context=spec.context) for child in value)

# -----------------------------------------------------------------------------
# Public geometry protocol
# -----------------------------------------------------------------------------

is_paramorph_type(::Type) = false
parameter_fields(::Type) = ()
auxiliary_fields(::Type) = ()
supports_type_geometry(::Type) = false
numeric_parameter_index(::Type) = nothing

function transformation_schema(T::Type, context::NamedTuple=NamedTuple())
    throw(ArgumentError("$T does not declare a Paramorph geometry"))
end
function transformation_schema(object, context::NamedTuple=NamedTuple())
    throw(ArgumentError("$(typeof(object)) does not declare a Paramorph geometry"))
end

function _schema_from_values(T::Type, values::NamedTuple, context::NamedTuple)
    throw(ArgumentError("$T does not declare a Paramorph geometry"))
end

function _schema_from_auxiliary(T::Type, auxiliary::NamedTuple, context::NamedTuple)
    throw(ArgumentError("$T does not declare a Paramorph geometry"))
end

function _parameter_values_from_values(T::Type, values::NamedTuple, context::NamedTuple)
    throw(ArgumentError("$T does not declare a Paramorph geometry"))
end

function _reconstruct_declared_from_type(T::Type, constrained::NamedTuple, context::NamedTuple)
    return _reconstruct_declared_from_type(T, constrained, context, NamedTuple())
end
function _reconstruct_declared_from_type(
    T::Type, constrained::NamedTuple, context::NamedTuple, auxiliary::NamedTuple,
)
    throw(ArgumentError("$T does not declare a Paramorph geometry"))
end
function _reconstruct_declared_from_prototype(
    prototype, T::Type, constrained::NamedTuple, context::NamedTuple,
)
    throw(ArgumentError("$(typeof(prototype)) does not declare a Paramorph geometry"))
end

function rebind_numeric_type(T::Type, ::Type{N}) where {N}
    index = numeric_parameter_index(T)
    index === nothing && return T
    U = Base.unwrap_unionall(T)
    parameters = collect(U.parameters)
    index <= length(parameters) || return T
    parameters[index] = N
    return Core.apply_type(Base.typename(U).wrapper, parameters...)
end

function _type_schema(T::Type, context::NamedTuple, auxiliary::NamedTuple)
    supports_type_geometry(T) || throw(ArgumentError(
        "$T has parameter geometry depending on constrained parameter values; use a prototype object",
    ))
    return isempty(auxiliary) ? transformation_schema(T, context) :
        _schema_from_auxiliary(T, auxiliary, context)
end

intrinsic_dimension(T::Type; context=NamedTuple(), auxiliary=NamedTuple()) =
    TransformVariables.dimension(_type_schema(T, context, auxiliary))
intrinsic_dimension(object; context=NamedTuple()) =
    TransformVariables.dimension(transformation_schema(object, context))

parameter_values(object; context=NamedTuple()) = begin
    values = NamedTuple{fieldnames(typeof(object))}(
        Tuple(getfield(object, field) for field in fieldnames(typeof(object))),
    )
    _parameter_values_from_values(typeof(object), values, context)
end

function constraint(
    T::Type, coordinates::AbstractVector{<:Real};
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

function constraint(prototype, coordinates::AbstractVector{<:Real}; context=NamedTuple())
    schema = transformation_schema(prototype, context)
    length(coordinates) == TransformVariables.dimension(schema) || throw(DimensionMismatch(
        "expected $(TransformVariables.dimension(schema)) coordinates, got $(length(coordinates))",
    ))
    target = rebind_numeric_type(typeof(prototype), eltype(coordinates))
    constrained = TransformVariables.transform(schema, coordinates)
    return _reconstruct_declared_from_prototype(prototype, target, constrained, context)
end

function constraint_with_logjac(
    T::Type, coordinates::AbstractVector{<:Real};
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

function constraint_with_logjac(prototype, coordinates::AbstractVector{<:Real}; context=NamedTuple())
    schema = transformation_schema(prototype, context)
    length(coordinates) == TransformVariables.dimension(schema) || throw(DimensionMismatch(
        "expected $(TransformVariables.dimension(schema)) coordinates, got $(length(coordinates))",
    ))
    constrained, logjac = TransformVariables.transform_and_logjac(schema, coordinates)
    target = rebind_numeric_type(typeof(prototype), eltype(coordinates))
    return _reconstruct_declared_from_prototype(prototype, target, constrained, context), logjac
end

function unconstrain(object; context=NamedTuple())
    schema = transformation_schema(object, context)
    return TransformVariables.inverse(schema, parameter_values(object; context))
end

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

function _validate_constrained(
    T::Type, values::NamedTuple, context::NamedTuple=NamedTuple(),
)
    schema = _schema_from_values(T, values, context)
    parameters = _parameter_values_from_values(T, values, context)
    try
        TransformVariables.inverse(schema, parameters)
    catch error
        error isa DomainError && rethrow()
        error isa DimensionMismatch && rethrow()
        error isa ArgumentError && rethrow()
        rethrow()
    end
    return nothing
end

# Internal token used by reconstruction to bypass user-facing validation after a
# transformation has already produced a value in the declared constrained space.
struct TrustedConstruction end
const _trusted_construction = TrustedConstruction()

# -----------------------------------------------------------------------------
# Macro parser helpers
# -----------------------------------------------------------------------------

function _type_parameter_parts(parameters)
    map(parameters) do parameter
        if parameter isa Symbol
            return parameter
        elseif parameter isa Expr && parameter.head in (:(<:), :(>:))
            return parameter.args[1]
        else
            error("unsupported type parameter declaration: $parameter")
        end
    end
end

function _field_name_from_decl(expr)
    expr isa Expr && expr.head == :(::) && expr.args[1] isa Symbol || return nothing
    return expr.args[1]
end

function _split_parameter_line(expr)
    expr isa Expr && expr.head == :call && expr.args[1] == :~ || return nothing
    length(expr.args) == 3 || error("parameter declaration must be `field::Type ~ geometry`")
    declaration, geometry = expr.args[2], expr.args[3]
    name = _field_name_from_decl(declaration)
    name === nothing && error("left-hand side of `~` must be a typed field declaration")
    return name, declaration.args[2], geometry
end

function _split_geometry_macro(expr)
    expr isa Expr && expr.head == :macrocall || return nothing
    expr.args[1] == Symbol("@geometry") || return nothing
    payload = expr.args[end]
    payload isa Expr && payload.head == :call && payload.args[1] == :~ ||
        error("@geometry expects `(fields...) ~ transform`")
    lhs, geometry = payload.args[2], payload.args[3]
    names = if lhs isa Symbol
        (lhs,)
    elseif lhs isa Expr && lhs.head == :tuple
        Tuple(lhs.args)
    else
        error("@geometry left-hand side must be a field name or tuple of field names")
    end
    all(name -> name isa Symbol, names) || error("@geometry fields must be symbols")
    return names, geometry
end

function _contains_field_reference(expr, fields::Set{Symbol})
    expr isa Symbol && return expr in fields
    expr isa QuoteNode && return false
    expr isa Expr || return false
    return any(arg -> _contains_field_reference(arg, fields), expr.args)
end

# -----------------------------------------------------------------------------
# @paramorph
# -----------------------------------------------------------------------------

"""
    @paramorph T struct Model{...,T<:Real,...}
        parameter::StorageType ~ geometry
        auxiliary::OtherType
    end

Attach a Paramorph parameter geometry to an immutable Julia struct. The storage
annotation after `::` is ordinary Julia; the expression after `~` is the
unconstrained-coordinate geometry for that field. Fields without `~` are
auxiliary and are excluded from the parameter vector.

Geometry expressions may reference type parameters, `context`, and fields of the
struct. Geometry depending on constrained parameter values requires a prototype;
geometry depending only on auxiliary fields can be supplied to type-based
operations with `auxiliary=(; field=value, ...)`.
"""
macro paramorph(numeric_parameter, expr)
    numeric_parameter isa Symbol || error("the numeric parameter passed to @paramorph must be a symbol")
    expr isa Expr && expr.head == :struct || error("@paramorph must wrap a struct definition")
    expr.args[1] && error("@paramorph does not support mutable structs")

    struct_sig = expr.args[2]
    declaration, supertype = if struct_sig isa Expr && struct_sig.head == :(<:)
        struct_sig.args[1], struct_sig.args[2]
    else
        struct_sig, nothing
    end

    if declaration isa Expr && declaration.head == :curly
        struct_name = declaration.args[1]
        type_params = collect(declaration.args[2:end])
    elseif declaration isa Symbol
        struct_name = declaration
        type_params = Any[]
    else
        error("unsupported struct declaration")
    end
    struct_name isa Symbol || error("unsupported struct name")

    type_args = _type_parameter_parts(type_params)
    numeric_index = findfirst(==(numeric_parameter), type_args)
    numeric_index === nothing && error(
        "numeric parameter $numeric_parameter must be explicitly declared by the struct",
    )
    numeric_parameter_decl = type_params[numeric_index]

    raw_lines = [line for line in expr.args[3].args if !(line isa LineNumberNode)]
    field_records = NamedTuple[]
    geometry_macro = nothing

    for line in raw_lines
        global_geometry = _split_geometry_macro(line)
        if global_geometry !== nothing
            geometry_macro === nothing || error("only one @geometry declaration is allowed")
            geometry_macro = global_geometry
            continue
        end

        parameter = _split_parameter_line(line)
        if parameter !== nothing
            name, storage, geometry = parameter
            push!(field_records, (; name, storage, geometry, is_parameter=true, default=nothing))
            continue
        end

        has_default = line isa Expr && line.head == :(=)
        declaration_line = has_default ? line.args[1] : line
        default = has_default ? line.args[2] : nothing
        name = _field_name_from_decl(declaration_line)
        name === nothing && error(
            "@paramorph struct bodies may contain only typed fields, `field::Type ~ geometry`, and @geometry",
        )
        push!(field_records, (;
            name,
            storage=declaration_line.args[2],
            geometry=nothing,
            is_parameter=false,
            default,
        ))
    end

    field_names = [r.name for r in field_records]
    length(unique(field_names)) == length(field_names) || error("duplicate field declaration")

    if geometry_macro !== nothing
        any(r -> r.is_parameter, field_records) && error(
            "do not mix field-level `~` declarations with a global @geometry declaration",
        )
        geometry_names, global_geometry = geometry_macro
        all(name -> name in field_names, geometry_names) || error("@geometry names an unknown field")
        field_records = [merge(r, (; is_parameter=r.name in geometry_names)) for r in field_records]
    else
        global_geometry = nothing
    end

    parameter_names = Tuple(r.name for r in field_records if r.is_parameter)
    auxiliary_names = Tuple(r.name for r in field_records if !r.is_parameter)
    isempty(parameter_names) && error("@paramorph requires at least one parameter field")

    clean_fields = [:( $(r.name)::$(r.storage) ) for r in field_records]
    all_values = Expr(:tuple, [:( $(r.name) = $(r.name) ) for r in field_records]...)
    defaults = Expr(:tuple, [:( $(r.name) = $(r.default) ) for r in field_records if !r.is_parameter && r.default !== nothing]...)

    field_set = Set{Symbol}(field_names)
    parameter_set = Set{Symbol}(parameter_names)
    auxiliary_set = Set{Symbol}(auxiliary_names)
    parameter_dependent = if global_geometry === nothing
        any(r -> r.is_parameter && _contains_field_reference(r.geometry, parameter_set), field_records)
    else
        _contains_field_reference(global_geometry, parameter_set)
    end
    auxiliary_dependent = if global_geometry === nothing
        any(r -> r.is_parameter && _contains_field_reference(r.geometry, auxiliary_set), field_records)
    else
        _contains_field_reference(global_geometry, auxiliary_set)
    end

    bind_from_values = [:( $(r.name) = getproperty(values, $(QuoteNode(r.name))) ) for r in field_records]

    auxiliary_bindings = Any[]
    for r in field_records
        r.is_parameter && continue
        fallback = r.default === nothing ?
            :(throw(ArgumentError(string(S, " needs auxiliary field ", $(string(r.name)), " for type-based parameter geometry")))) :
            r.default
        push!(auxiliary_bindings, :(
            $(r.name) = hasproperty(auxiliary, $(QuoteNode(r.name))) ?
                getproperty(auxiliary, $(QuoteNode(r.name))) : $fallback
        ))
    end

    parameter_records = [r for r in field_records if r.is_parameter]

    function field_schema_expr(r, value_expr, have_value)
        return :(Paramorph._materialize_geometry(
            $(r.geometry),
            fieldtype(S, $(QuoteNode(r.name))),
            $value_expr,
            $have_value,
        ))
    end

    if global_geometry === nothing
        value_schema_pairs = [
            Expr(:(=), r.name, field_schema_expr(r, :(getproperty(values, $(QuoteNode(r.name)))), true))
            for r in parameter_records
        ]
        type_schema_pairs = [
            Expr(:(=), r.name, field_schema_expr(r, nothing, false))
            for r in parameter_records
        ]
        value_schema_expr = :(TransformVariables.as(($(value_schema_pairs...),)))
        type_schema_expr = :(TransformVariables.as(($(type_schema_pairs...),)))
    else
        value_schema_expr = global_geometry
        type_schema_expr = global_geometry
    end

    parameter_value_pairs = Any[]
    type_reconstructed_parameters = Dict{Symbol,Any}()
    prototype_reconstructed_parameters = Dict{Symbol,Any}()

    if global_geometry === nothing
        for r in parameter_records
            name, geometry = r.name, r.geometry
            value_expr = :(getproperty(values, $(QuoteNode(name))))
            parameter_value = :(
                Paramorph._parameter_value($geometry, $value_expr, context)
            )
            push!(parameter_value_pairs, Expr(:(=), name, parameter_value))
            type_reconstructed_parameters[name] = :(
                Paramorph._reconstruct_parameter(
                    $geometry,
                    fieldtype(S, $(QuoteNode(name))),
                    getproperty(constrained, $(QuoteNode(name))),
                    nothing,
                    false,
                )
            )
            prototype_reconstructed_parameters[name] = :(
                Paramorph._reconstruct_parameter(
                    $geometry,
                    fieldtype(S, $(QuoteNode(name))),
                    getproperty(constrained, $(QuoteNode(name))),
                    getproperty(prototype, $(QuoteNode(name))),
                    true,
                )
            )
        end
    else
        parameter_value_pairs = [
            Expr(:(=), name, :(getproperty(values, $(QuoteNode(name)))))
            for name in parameter_names
        ]
        for name in parameter_names
            type_reconstructed_parameters[name] = :(getproperty(constrained, $(QuoteNode(name))))
            prototype_reconstructed_parameters[name] = :(getproperty(constrained, $(QuoteNode(name))))
        end
    end

    value_parameter_expr = :(($(parameter_value_pairs...),))

    type_schema_body = if parameter_dependent
        :(throw(ArgumentError(string(
            S, " has parameter geometry depending on constrained parameter values; use a prototype object",
        ))))
    elseif auxiliary_dependent
        :(throw(ArgumentError(string(
            S, " has parameter geometry depending on auxiliary field values; pass `auxiliary=(; ...)` or use a prototype object",
        ))))
    else
        type_schema_expr
    end

    default_auxiliary_values = Dict{Symbol,Any}(
        r.name => r.default for r in field_records if !r.is_parameter && r.default !== nothing
    )

    direct_type_field_values = Any[]
    auxiliary_type_field_values = Any[]
    prototype_field_values = Any[]
    for r in field_records
        if r.is_parameter
            type_value = get(type_reconstructed_parameters, r.name, :(getproperty(constrained, $(QuoteNode(r.name)))))
            proto_value = get(prototype_reconstructed_parameters, r.name, :(getproperty(constrained, $(QuoteNode(r.name)))))
            push!(direct_type_field_values, type_value)
            push!(auxiliary_type_field_values, type_value)
            push!(prototype_field_values, proto_value)
        else
            direct_default = get(default_auxiliary_values, r.name, :(
                throw(ArgumentError(string(
                    S, " needs auxiliary field ", $(string(r.name)), " for type-based reconstruction",
                )))
            ))
            push!(direct_type_field_values, direct_default)
            push!(auxiliary_type_field_values, r.name)
            push!(prototype_field_values, :(getproperty(prototype, $(QuoteNode(r.name)))))
        end
    end

    struct_body = Expr(:block, clean_fields...)
    struct_expr = supertype === nothing ?
        Expr(:struct, false, declaration, struct_body) :
        Expr(:struct, false, Expr(:(<:), declaration, supertype), struct_body)

    methods = quote
        Paramorph.is_paramorph_type(::Type{<:$struct_name}) = true
        Paramorph.parameter_fields(::Type{<:$struct_name}) = $(QuoteNode(parameter_names))
        Paramorph.auxiliary_fields(::Type{<:$struct_name}) = $(QuoteNode(auxiliary_names))
        Paramorph.supports_type_geometry(::Type{<:$struct_name}) = $(!parameter_dependent)
        Paramorph.numeric_parameter_index(::Type{<:$struct_name}) = $numeric_index

        function Paramorph.transformation_schema(
            ::Type{S}, context::NamedTuple=NamedTuple(),
        ) where {$(type_params...), S<:$struct_name{$(type_args...)}}
            return $type_schema_body
        end

        function Paramorph._schema_from_auxiliary(
            ::Type{S}, auxiliary::NamedTuple, context::NamedTuple,
        ) where {$(type_params...), S<:$struct_name{$(type_args...)}}
            $parameter_dependent && throw(ArgumentError(string(
                S, " has parameter geometry depending on constrained parameter values; use a prototype object",
            )))
            $(auxiliary_bindings...)
            return $type_schema_expr
        end

        function Paramorph._schema_from_values(
            ::Type{S}, values::NamedTuple, context::NamedTuple,
        ) where {$(type_params...), S<:$struct_name{$(type_args...)}}
            $(bind_from_values...)
            return $value_schema_expr
        end

        function Paramorph.transformation_schema(
            object::S, context::NamedTuple=NamedTuple(),
        ) where {$(type_params...), S<:$struct_name{$(type_args...)}}
            values = NamedTuple{fieldnames(S)}(
                Tuple(getfield(object, field) for field in fieldnames(S)),
            )
            return Paramorph._schema_from_values(S, values, context)
        end

        function Paramorph._parameter_values_from_values(
            ::Type{S}, values::NamedTuple, context::NamedTuple,
        ) where {$(type_params...), S<:$struct_name{$(type_args...)}}
            $(bind_from_values...)
            return $value_parameter_expr
        end

        function Paramorph._reconstruct_declared_from_type(
            ::Type{S}, constrained::NamedTuple, context::NamedTuple,
        ) where {$(type_params...), S<:$struct_name{$(type_args...)}}
            return Paramorph._reconstruct_declared_from_type(
                S, constrained, context, NamedTuple(),
            )
        end

        function Paramorph._reconstruct_declared_from_type(
            ::Type{S}, constrained::NamedTuple, context::NamedTuple, auxiliary::NamedTuple,
        ) where {$(type_params...), S<:$struct_name{$(type_args...)}}
            $parameter_dependent && throw(ArgumentError(string(
                S, " has value-dependent parameter geometry; use a prototype object",
            )))
            $(auxiliary_bindings...)
            return S(Paramorph._trusted_construction, $(auxiliary_type_field_values...))
        end

        function Paramorph._reconstruct_declared_from_prototype(
            prototype::$struct_name, ::Type{S}, constrained::NamedTuple, context::NamedTuple,
        ) where {$(type_params...), S<:$struct_name{$(type_args...)}}
            return S(Paramorph._trusted_construction, $(prototype_field_values...))
        end

        function $struct_name{$(type_args...)}(
            ::Paramorph.TrustedConstruction,
            $(clean_fields...),
        ) where {$(type_params...)}
            return new{$(type_args...)}($(r.name for r in field_records...))
        end

        function $struct_name{$(type_args...)}($(clean_fields...)) where {$(type_params...)}
            values = $all_values
            Paramorph._validate_constrained($struct_name{$(type_args...)}, values)
            return $struct_name{$(type_args...)}(Paramorph._trusted_construction, $(r.name for r in field_records...))
        end
    end

    return esc(quote
        $struct_expr
        $methods
    end)
end

end
