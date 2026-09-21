from pathlib import Path

path = Path("src/Paramorph.jl")
text = path.read_text()

old = '''function _schema_from_values(T::Type, values::NamedTuple, context::NamedTuple)\n    throw(ArgumentError("$T does not declare a Paramorph geometry"))\nend\n\nfunction _parameter_values_from_values(T::Type, values::NamedTuple, context::NamedTuple)\n'''
new = '''function _schema_from_values(T::Type, values::NamedTuple, context::NamedTuple)\n    throw(ArgumentError("$T does not declare a Paramorph geometry"))\nend\n\nfunction _schema_from_auxiliary(T::Type, auxiliary::NamedTuple, context::NamedTuple)\n    throw(ArgumentError("$T does not declare a Paramorph geometry"))\nend\n\nfunction _parameter_values_from_values(T::Type, values::NamedTuple, context::NamedTuple)\n'''
assert old in text
text = text.replace(old, new, 1)

old = '''_reconstruct_declared_from_type(T::Type, constrained::NamedTuple, context::NamedTuple) =\n    _reconstruct_from_type(T, constrained)\n_reconstruct_declared_from_prototype(prototype, T::Type, constrained::NamedTuple, context::NamedTuple) =\n    _reconstruct_from_prototype(prototype, T, constrained)\n'''
new = '''_reconstruct_declared_from_type(T::Type, constrained::NamedTuple, context::NamedTuple) =\n    _reconstruct_declared_from_type(T, constrained, context, NamedTuple())\n_reconstruct_declared_from_type(T::Type, constrained::NamedTuple, context::NamedTuple, auxiliary::NamedTuple) =\n    _reconstruct_from_type(T, constrained)\n_reconstruct_declared_from_prototype(prototype, T::Type, constrained::NamedTuple, context::NamedTuple) =\n    _reconstruct_from_prototype(prototype, T, constrained)\n'''
assert old in text
text = text.replace(old, new, 1)

start = text.index('intrinsic_dimension(T::Type; context=NamedTuple()) =')
end = text.index('\n# -----------------------------------------------------------------------------\n# Validation', start)
old = text[start:end]
new = r'''function _type_schema(T::Type, context::NamedTuple, auxiliary::NamedTuple)
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
    values = NamedTuple{fieldnames(typeof(object))}(Tuple(getfield(object, field) for field in fieldnames(typeof(object))))
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
'''
text = text[:start] + new + text[end:]

old = '''    field_set = Set(field_names)\n    runtime_dependent = if global_geometry === nothing\n        any(r -> r.is_parameter && _contains_field_reference(r.geometry, field_set), field_records)\n    else\n        _contains_field_reference(global_geometry, field_set)\n    end\n\n    # Bind every field name while evaluating runtime geometry expressions.\n    bind_from_values = [:( $(r.name) = getproperty(values, $(QuoteNode(r.name))) ) for r in field_records]\n'''
new = '''    field_set = Set(field_names)\n    parameter_set = Set(parameter_names)\n    auxiliary_set = Set(auxiliary_names)\n    parameter_dependent = if global_geometry === nothing\n        any(r -> r.is_parameter && _contains_field_reference(r.geometry, parameter_set), field_records)\n    else\n        _contains_field_reference(global_geometry, parameter_set)\n    end\n    auxiliary_dependent = if global_geometry === nothing\n        any(r -> r.is_parameter && _contains_field_reference(r.geometry, auxiliary_set), field_records)\n    else\n        _contains_field_reference(global_geometry, auxiliary_set)\n    end\n\n    # Bind every field name while evaluating geometry from a concrete object.\n    bind_from_values = [:( $(r.name) = getproperty(values, $(QuoteNode(r.name))) ) for r in field_records]\n\n    # Type-based geometry may use stored auxiliary fields when the caller supplies\n    # them explicitly. Defaults remain available for auxiliary fields that have one.\n    auxiliary_bindings = Any[]\n    for r in field_records\n        r.is_parameter && continue\n        fallback = r.default === nothing ?\n            :(throw(ArgumentError(string(S, " needs auxiliary field ", $(string(r.name)), " for type-based parameter geometry")))) :\n            r.default\n        push!(auxiliary_bindings, :(\n            $(r.name) = hasproperty(auxiliary, $(QuoteNode(r.name))) ?\n                getproperty(auxiliary, $(QuoteNode(r.name))) : $fallback\n        ))\n    end\n'''
assert old in text
text = text.replace(old, new, 1)

old = '''    type_schema_body = runtime_dependent ? :(throw(ArgumentError(string(\n        S, " has parameter geometry depending on runtime field values; use a prototype object",\n    )))) : type_schema_expr\n'''
new = '''    type_schema_body = if parameter_dependent\n        :(throw(ArgumentError(string(\n            S, " has parameter geometry depending on constrained parameter values; use a prototype object",\n        ))))\n    elseif auxiliary_dependent\n        :(throw(ArgumentError(string(\n            S, " has parameter geometry depending on auxiliary field values; pass `auxiliary=(; ...)` or use a prototype object",\n        ))))\n    else\n        type_schema_expr\n    end\n'''
assert old in text
text = text.replace(old, new, 1)

old = '''        Paramorph.supports_type_geometry(::Type{<:$struct_name}) = $(!runtime_dependent)\n'''
new = '''        Paramorph.supports_type_geometry(::Type{<:$struct_name}) = $(!parameter_dependent)\n'''
assert old in text
text = text.replace(old, new, 1)

old = '''        function Paramorph.transformation_schema(\n            ::Type{S}, context::NamedTuple=NamedTuple(),\n        ) where {$(type_params...), S<:$struct_name{$(type_args...)}}\n            return $type_schema_body\n        end\n\n        function Paramorph._parameter_values_from_values(\n'''
new = '''        function Paramorph.transformation_schema(\n            ::Type{S}, context::NamedTuple=NamedTuple(),\n        ) where {$(type_params...), S<:$struct_name{$(type_args...)}}\n            return $type_schema_body\n        end\n\n        function Paramorph._schema_from_auxiliary(\n            ::Type{S}, auxiliary::NamedTuple, context::NamedTuple,\n        ) where {$(type_params...), S<:$struct_name{$(type_args...)}}\n            $parameter_dependent && throw(ArgumentError(string(\n                S, " has parameter geometry depending on constrained parameter values; use a prototype object",\n            )))\n            $(auxiliary_bindings...)\n            return $type_schema_expr\n        end\n\n        function Paramorph._parameter_values_from_values(\n'''
assert old in text
text = text.replace(old, new, 1)

old = '''        function Paramorph._reconstruct_declared_from_type(\n            ::Type{S}, constrained::NamedTuple, context::NamedTuple,\n        ) where {$(type_params...), S<:$struct_name{$(type_args...)}}\n            $(runtime_dependent ? :(throw(ArgumentError(string(\n                S, " has value-dependent parameter geometry; use a prototype object",\n            )))) : nothing)\n            return S(Paramorph._trusted_construction, $(type_field_values...))\n        end\n'''
new = '''        function Paramorph._reconstruct_declared_from_type(\n            ::Type{S}, constrained::NamedTuple, context::NamedTuple,\n        ) where {$(type_params...), S<:$struct_name{$(type_args...)}}\n            return Paramorph._reconstruct_declared_from_type(\n                S, constrained, context, NamedTuple(),\n            )\n        end\n\n        function Paramorph._reconstruct_declared_from_type(\n            ::Type{S}, constrained::NamedTuple, context::NamedTuple, auxiliary::NamedTuple,\n        ) where {$(type_params...), S<:$struct_name{$(type_args...)}}\n            $parameter_dependent && throw(ArgumentError(string(\n                S, " has value-dependent parameter geometry; use a prototype object",\n            )))\n            $(auxiliary_bindings...)\n            return S(Paramorph._trusted_construction, $(\n                [r.is_parameter ? get(type_reconstructed_parameters, r.name, :(getproperty(constrained, $(QuoteNode(r.name))))) : r.name for r in field_records]...\n            ))\n        end\n'''
assert old in text
text = text.replace(old, new, 1)

path.write_text(text)
print("auxiliary-aware type geometry patch applied")
