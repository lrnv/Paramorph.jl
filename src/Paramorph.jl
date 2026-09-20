module Paramorph

import TransformVariables
using LinearAlgebra

export @paramorph, transformation_schema, constraint, unconstrain, dimension_intrinsique, constraint_with_logjac

# Default transformation for unconstrained scalar fields.
transformation_schema(T::Type) = TransformVariables.asℝ

"""
    @paramorph struct MyStruct{T, N}
        a::T::asℝ₊
        b::Vector{T}::UnitSimplex(N)
    end

Define a structure whose fields are validated and transformed by TransformVariables.jl.
"""
macro paramorph(expr)
    if expr.head != :struct
        error("@paramorph must be applied to a struct definition")
    end
    
    # A `struct` expression is `(mutable?, signature, body)`.
    struct_sig = expr.args[2]
    
    # Extract the structure name and its type parameters.
    if struct_sig isa Expr && struct_sig.head == :curly
        struct_name = struct_sig.args[1]
        type_params = struct_sig.args[2:end]
    else
        struct_name = struct_sig
        type_params = []
    end
    type_args = map(type_params) do parameter
        parameter isa Symbol && return parameter
        parameter isa Expr && parameter.head in (:<:, :>:) && return parameter.args[1]
        error("Unsupported type parameter: $parameter")
    end
    
    body = expr.args[3].args
    
    clean_fields = []
    schema_pairs = []
    field_names = []
    
    for line in body
        if line isa Expr && line.head == :(::)
            
            # Explicit transform: b::Vector{T}::UnitSimplex(N)
            if line.args[1] isa Expr && line.args[1].head == :(::)
                inner_expr = line.args[1]
                field_name = inner_expr.args[1]
                field_type = inner_expr.args[2]
                constraint = line.args[2]
                
                push!(field_names, field_name)
                push!(clean_fields, :($field_name::$field_type))
                push!(schema_pairs, :($field_name = $constraint))
                
            # No explicit transform: recursively use the field type's schema.
            else
                field_name = line.args[1]
                field_type = line.args[2]
                
                push!(field_names, field_name)
                push!(clean_fields, :($field_name::$field_type))
                push!(schema_pairs, :($field_name = transformation_schema($field_type)))
            end
        end
    end

    schema_namedtuple = Expr(:tuple, schema_pairs...)
    values_namedtuple = Expr(:tuple, [:( $name = $name ) for name in field_names]...)
    constructor = if isempty(type_params)
        quote
            function $struct_name($(clean_fields...))
                Paramorph.validate_constrained($struct_name, $values_namedtuple)
                return new($(field_names...))
            end
        end
    else
        quote
            function $struct_name{$(type_args...)}($(clean_fields...)) where {$(type_params...)}
                Paramorph.validate_constrained(
                    $struct_name{$(type_args...)}, $values_namedtuple
                )
                return new{$(type_args...)}($(field_names...))
            end
        end
    end
    
    # Generate the structure, its only inner constructor, and its schema.
    return esc(quote
        struct $struct_sig
            $(clean_fields...)
            $constructor
        end
        
        # Capture value type parameters such as N in transformation expressions.
        function Paramorph.transformation_schema(::Type{S}) where {$(type_params...), S<:$struct_name{$(type_params...)}}
            return Paramorph.TransformVariables.as($schema_namedtuple)
        end

        function Paramorph.reconstruct_struct(::Type{S}, nt::NamedTuple) where {$(type_params...), S<:$struct_name{$(type_params...)}}
            args = map(fieldnames(S)) do field
                Paramorph.reconstruct_field(fieldtype(S, field), getproperty(nt, field))
            end
            return S(args...)
        end
    end)
end

function _same_constrained(a::NamedTuple, b::NamedTuple)
    keys(a) == keys(b) || return false
    return all(_same_constrained(a[name], b[name]) for name in keys(a))
end

function _same_constrained(a::AbstractArray, b::AbstractArray)
    axes(a) == axes(b) || return false
    return all(_same_constrained(x, y) for (x, y) in zip(a, b))
end

_same_constrained(a::Number, b::Number) = isapprox(a, b)
_same_constrained(a, b) = isequal(a, b)

function validate_constrained(T::Type, values::NamedTuple)
    schema = transformation_schema(T)
    plain_values = to_named_tuple(values)
    coordinates = try
        TransformVariables.inverse(schema, plain_values)
    catch error
        error isa Union{DomainError, ArgumentError, DimensionMismatch} || rethrow()
        throw(DomainError(values, "the fields of $T do not satisfy their constraints"))
    end
    roundtrip = TransformVariables.transform(schema, coordinates)
    _same_constrained(plain_values, roundtrip) || throw(DomainError(
        values,
        "the fields of $T are not in the image of their transformation",
    ))
    return nothing
end

reconstruct_field(T::Type, value::NamedTuple) = reconstruct_struct(T, value)
reconstruct_field(::Type, value) = value
reconstruct_struct(::Type, value) = value

function to_named_tuple(obj)
    names = fieldnames(typeof(obj))
    values = map(names) do field
        value = getproperty(obj, field)
        value isa AbstractArray || isprimitivetype(typeof(value)) || value isa String ?
            value : to_named_tuple(value)
    end
    return NamedTuple{names}(values)
end

# Public API

dimension_intrinsique(T::Type) = TransformVariables.dimension(transformation_schema(T))

function constraint(T::Type, x::Vector{<:Real})
    schema = transformation_schema(T)
    @assert length(x) == TransformVariables.dimension(schema) "Incorrect vector length."
    return reconstruct_struct(T, TransformVariables.transform(schema, x))
end

function constraint_with_logjac(T::Type, x::Vector{<:Real})
    schema = transformation_schema(T)
    @assert length(x) == TransformVariables.dimension(schema) "Incorrect vector length."
    nt, logjac = TransformVariables.transform_and_logjac(schema, x)
    return reconstruct_struct(T, nt), logjac
end

function unconstrain(obj)
    schema = transformation_schema(typeof(obj))
    return TransformVariables.inverse(schema, to_named_tuple(obj))
end

end # module
