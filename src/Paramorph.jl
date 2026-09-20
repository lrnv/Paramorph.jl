module Paramorph

import TransformVariables
using LinearAlgebra

export @paramorph, transformed_type, transformation_schema, constraint, unconstrain, dimension_intrinsique, constraint_with_logjac

# Default transformation for unconstrained scalar fields.
transformation_schema(T::Type) = TransformVariables.asℝ
is_paramorph_type(::Type) = false
parameter_fields(::Type) = ()
auxiliary_fields(::Type) = ()
auxiliary_defaults(::Type) = NamedTuple()

const _PARAMORPH_TYPES = Set{Tuple{Module,Symbol}}()

function _annotation_name(expr)
    expr isa Symbol && return expr
    expr isa Expr && expr.head == :curly && return _annotation_name(expr.args[1])
    return nothing
end

_is_paramorph_annotation(expr, module_) =
    (_annotation_name(expr) isa Symbol) && ((module_, _annotation_name(expr)) in _PARAMORPH_TYPES)

"""
    transformed_type(transformation, T)

Return the constrained value type produced by `transformation` from coordinates
with element type `T`. Custom transformations can extend this function.
"""
transformed_type(t::TransformVariables.ScalarTransform, ::Type{T}) where {T} =
    typeof(TransformVariables.transform(t, zero(T)))

function transformed_type(t::TransformVariables.ArrayTransformation{<:Any,M}, ::Type{T}) where {M,T}
    E = transformed_type(t.inner_transformation, T)
    return Array{E,M}
end

transformed_type(::TransformVariables.ViewTransformation{M}, ::Type{T}) where {M,T} =
    AbstractArray{T,M}
transformed_type(::TransformVariables.UnitVector, ::Type{T}) where {T} = Vector{T}
transformed_type(::TransformVariables.UnitSimplex, ::Type{T}) where {T} = Vector{T}
transformed_type(::TransformVariables.UnitVectorNorm, ::Type{T}) where {T} = Tuple{Vector{T},T}
transformed_type(::TransformVariables.CorrCholeskyFactor, ::Type{T}) where {T} =
    UpperTriangular{T,Matrix{T}}
transformed_type(t::TransformVariables.Constant, ::Type) = typeof(t.value)
transformed_type(::TransformVariables.TypeWrapperTransform{S}, ::Type) where {S} = S

function transformed_type(t::TransformVariables.TransformTuple, ::Type{T}) where {T}
    inner = getfield(t, :inner)
    types = map(transform -> transformed_type(transform, T), values(inner))
    tuple_type = Core.apply_type(Tuple, types...)
    return inner isa NamedTuple ? Core.apply_type(NamedTuple, keys(inner), tuple_type) : tuple_type
end

function transformed_type(t::TransformVariables.AbstractTransform, ::Type{T}) where {T}
    coordinates = zeros(T, TransformVariables.dimension(t))
    return typeof(TransformVariables.transform(t, coordinates))
end

_transform_head(expr::Symbol) = expr
_transform_head(expr::Expr) = expr.head == :. ? expr.args[end] :
    (expr.head == :call ? _transform_head(expr.args[1]) : nothing)
_transform_head(::Any) = nothing

const _SCALAR_TRANSFORM_NAMES = Set((
    :asℝ, :asℝ₊, :asℝ₋, :as𝕀, :as_real, :as_positive_real,
    :as_negative_real, :Identity, :TVExp, :TVScale, :TVShift, :TVLogistic, :TVNeg,
))

function _is_transform_expr(expr)
    expr isa Expr && expr.head == :call && return true
    name = _transform_head(expr)
    return name in _SCALAR_TRANSFORM_NAMES || name in (
        :as, :UnitVector, :unit_vector_norm, :UnitSimplex,
        :CorrCholeskyFactor, :corr_cholesky_factor, :Constant, :CustomTransform,
    )
end

function _storage_type_expr(expr, numeric_type)
    name = _transform_head(expr)
    name in _SCALAR_TRANSFORM_NAMES && return numeric_type
    name in (:UnitVector, :UnitSimplex) && return :(Vector{$numeric_type})
    name == :unit_vector_norm && return :(Tuple{Vector{$numeric_type}, $numeric_type})
    name in (:CorrCholeskyFactor, :corr_cholesky_factor) &&
        return :(Paramorph.LinearAlgebra.UpperTriangular{$numeric_type, Matrix{$numeric_type}})
    name == :Constant && return :(typeof($(expr.args[2])))
    name == :CustomTransform && error(
        "@paramorph cannot infer the output type of CustomTransform; wrap it in a named custom rule",
    )

    if name == :as && expr isa Expr && expr.head == :call
        target = expr.args[2]
        target_name = _transform_head(target)
        offset = length(expr.args) >= 4 && _is_transform_expr(expr.args[3]) ? 1 : 0
        element_transform = offset == 1 ? expr.args[3] : :asℝ
        element_type = _storage_type_expr(element_transform, numeric_type)
        target_name == :Vector && return :(Vector{$element_type})
        target_name == :Matrix && return :(Matrix{$element_type})
        if target_name == :Array
            ndims = length(expr.args) - 2 - offset
            return :(Array{$element_type,$ndims})
        end
    end
    error("@paramorph has no output-type rule for transformation `$expr`")
end

"""
    @paramorph struct MyStruct{N}
        a::asℝ₊
        b::UnitSimplex(N)
    end

Define a structure whose fields are validated and transformed by TransformVariables.jl.
"""
macro paramorph(expr)
    if expr.head != :struct
        error("@paramorph must be applied to a struct definition")
    end
    expr.args[1] && error("@paramorph does not support mutable structs because mutation could bypass validation")
    
    # A `struct` expression is `(mutable?, signature, body)`.
    struct_sig = expr.args[2]
    declaration, supertype = if struct_sig isa Expr && struct_sig.head == :(<:)
        struct_sig.args[1], struct_sig.args[2]
    else
        struct_sig, nothing
    end
    
    # Extract user type parameters. T is reserved and appended automatically.
    if declaration isa Expr && declaration.head == :curly
        struct_name = declaration.args[1]
        user_type_params = declaration.args[2:end]
    else
        struct_name = declaration
        user_type_params = []
    end
    user_type_args = map(user_type_params) do parameter
        parameter isa Symbol && return parameter
        parameter isa Expr && parameter.head in (:<:, :>:) && return parameter.args[1]
        error("Unsupported type parameter: $parameter")
    end
    :T in user_type_args && error("T is reserved by @paramorph and must not be declared explicitly")
    type_params = [user_type_params..., :(T<:Real)]
    type_args = map(type_params) do parameter
        parameter isa Symbol && return parameter
        parameter isa Expr && parameter.head in (:<:, :>:) && return parameter.args[1]
        error("Unsupported type parameter: $parameter")
    end
    generated_declaration = Expr(:curly, struct_name, type_params...)
    generated_struct_sig = isnothing(supertype) ? generated_declaration :
        Expr(:(<:), generated_declaration, supertype)
    
    body = expr.args[3].args
    
    clean_fields = []
    schema_pairs = []
    field_names = []
    parameter_field_names = []
    auxiliary_field_names = []
    auxiliary_default_pairs = []
    
    for line in body
        line isa LineNumberNode && continue
        has_default = line isa Expr && line.head == :(=)
        declaration_line = has_default ? line.args[1] : line
        default_value = has_default ? line.args[2] : nothing
        if declaration_line isa Expr && declaration_line.head == :(::)
            
            field_name = declaration_line.args[1]
            annotation = declaration_line.args[2]
            field_name isa Symbol || error("@paramorph fields must use the form `name::transformation`")

            if _is_transform_expr(annotation)
                has_default && error("transformed field $field_name cannot have a default value")
                field_type = _storage_type_expr(annotation, :T)
                push!(field_names, field_name)
                push!(parameter_field_names, field_name)
                push!(clean_fields, :($field_name::$field_type))
                push!(schema_pairs, :($field_name = $annotation))
            elseif _is_paramorph_annotation(annotation, __module__)
                has_default && error("nested Paramorph field $field_name cannot have a default value")
                push!(field_names, field_name)
                push!(parameter_field_names, field_name)
                push!(clean_fields, :($field_name::$annotation))
                push!(schema_pairs, :($field_name = transformation_schema($annotation)))
            else
                push!(field_names, field_name)
                push!(auxiliary_field_names, field_name)
                push!(clean_fields, :($field_name::$annotation))
                has_default && push!(auxiliary_default_pairs, :($field_name = $default_value))
            end
        else
            error("@paramorph only accepts typed field declarations; unsupported entry: $line")
        end
    end

    schema_namedtuple = Expr(:tuple, schema_pairs...)
    defaults_namedtuple = Expr(:tuple, auxiliary_default_pairs...)
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
    inferred_constructor = if isempty(user_type_params)
        quote
            function $struct_name($(clean_fields...)) where {T}
                return $struct_name{T}($(field_names...))
            end
        end
    else
        quote
            function $struct_name{$(user_type_args...)}($(clean_fields...)) where {$(user_type_params...), T}
                return $struct_name{$(user_type_args...), T}($(field_names...))
            end
        end
    end
    generated_body = Expr(:block, clean_fields..., constructor.args...)
    struct_definition = Expr(:struct, false, generated_struct_sig, generated_body)
    push!(_PARAMORPH_TYPES, (__module__, struct_name))
    
    # Generate the structure, its only inner constructor, and its schema.
    return esc(quote
        $struct_definition

        $inferred_constructor
        
        # Capture value type parameters such as N in transformation expressions.
        function Paramorph.transformation_schema(::Type{S}) where {$(type_params...), S<:$struct_name{$(type_args...)}}
            return Paramorph.TransformVariables.as($schema_namedtuple)
        end

        Paramorph.is_paramorph_type(::Type{<:$struct_name}) = true
        Paramorph.parameter_fields(::Type{<:$struct_name}) = $(Tuple(parameter_field_names))
        Paramorph.auxiliary_fields(::Type{<:$struct_name}) = $(Tuple(auxiliary_field_names))
        function Paramorph.auxiliary_defaults(::Type{S}) where {$(type_params...), S<:$struct_name{$(type_args...)}}
            return $defaults_namedtuple
        end

        function Paramorph.reconstruct_struct(::Type{S}, nt::NamedTuple) where {$(type_params...), S<:$struct_name{$(type_args...)}}
            defaults = Paramorph.auxiliary_defaults(S)
            args = map(fieldnames(S)) do field
                if hasproperty(nt, field)
                    Paramorph.reconstruct_field(fieldtype(S, field), getproperty(nt, field))
                elseif hasproperty(defaults, field)
                    getproperty(defaults, field)
                else
                    throw(ArgumentError(
                        "$S has no default for auxiliary field $field; call constraint with a prototype object",
                    ))
                end
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
    plain_values = to_parameter_named_tuple(T, values)
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
reconstruct_struct(::Type, value::NamedTuple) = value
reconstruct_struct(::Type, value) = value

function to_parameter_named_tuple(T::Type, values::NamedTuple)
    names = parameter_fields(T)
    converted = map(names) do field
        value = getproperty(values, field)
        is_paramorph_type(typeof(value)) ? to_parameter_named_tuple(value) : value
    end
    return NamedTuple{names}(converted)
end

function to_parameter_named_tuple(obj)
    T = typeof(obj)
    names = parameter_fields(T)
    values = map(names) do field
        value = getproperty(obj, field)
        is_paramorph_type(typeof(value)) ? to_parameter_named_tuple(value) : value
    end
    return NamedTuple{names}(values)
end

function reconstruct_struct(prototype, nt::NamedTuple)
    T = typeof(prototype)
    is_paramorph_type(T) || return nt
    args = map(fieldnames(T)) do field
        old = getproperty(prototype, field)
        hasproperty(nt, field) ? reconstruct_field(old, getproperty(nt, field)) : old
    end
    return T(args...)
end

reconstruct_field(prototype, value::NamedTuple) = reconstruct_struct(prototype, value)
reconstruct_field(::Any, value) = value

# Public API

dimension_intrinsique(T::Type) = TransformVariables.dimension(transformation_schema(T))

function _check_coordinate_type(T::Type, x::Vector)
    T isa DataType || throw(ArgumentError("a concrete @paramorph type is required"))
    expected = T.parameters[end]
    eltype(x) == expected || throw(ArgumentError(
        "coordinate element type $(eltype(x)) does not match $T, which expects $expected",
    ))
end

function constraint(T::Type, x::Vector{<:Real})
    _check_coordinate_type(T, x)
    schema = transformation_schema(T)
    @assert length(x) == TransformVariables.dimension(schema) "Incorrect vector length."
    return reconstruct_struct(T, TransformVariables.transform(schema, x))
end

function constraint(prototype, x::Vector{<:Real})
    T = typeof(prototype)
    _check_coordinate_type(T, x)
    schema = transformation_schema(T)
    @assert length(x) == TransformVariables.dimension(schema) "Incorrect vector length."
    return reconstruct_struct(prototype, TransformVariables.transform(schema, x))
end

function constraint_with_logjac(T::Type, x::Vector{<:Real})
    _check_coordinate_type(T, x)
    schema = transformation_schema(T)
    @assert length(x) == TransformVariables.dimension(schema) "Incorrect vector length."
    nt, logjac = TransformVariables.transform_and_logjac(schema, x)
    return reconstruct_struct(T, nt), logjac
end

function constraint_with_logjac(prototype, x::Vector{<:Real})
    T = typeof(prototype)
    _check_coordinate_type(T, x)
    schema = transformation_schema(T)
    @assert length(x) == TransformVariables.dimension(schema) "Incorrect vector length."
    nt, logjac = TransformVariables.transform_and_logjac(schema, x)
    return reconstruct_struct(prototype, nt), logjac
end

function unconstrain(obj)
    schema = transformation_schema(typeof(obj))
    return TransformVariables.inverse(schema, to_parameter_named_tuple(obj))
end

end # module
