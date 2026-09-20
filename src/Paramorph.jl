module Paramorph

using TransformVariables
using LinearAlgebra

export @constrained_struct, transformation_schema, constraint, unconstrain, dimension_intrinsique, constraint_with_logjac

# Comportement racine par défaut
transformation_schema(T::Type) = asℝ

"""
    @constrained_struct struct MyStruct{T, N}
        a::T::asℝ₊
        b::Vector{T}::UnitSimplex(N)
    end

Macro universelle gérant les contraintes statiques et dynamiques (liées aux paramètres de type).
"""
macro constrained_struct(expr)
    if expr.head != :struct
        error("@constrained_struct doit être appliqué sur une structure (struct)")
    end
    
    # A `struct` expression is `(mutable?, signature, body)`.
    struct_sig = expr.args[2]
    
    # Extraction propre du nom et des paramètres {T, N, ...}
    if struct_sig isa Expr && struct_sig.head == :curly
        struct_name = struct_sig.args[1]
        type_params = struct_sig.args[2:end]
    else
        struct_name = struct_sig
        type_params = []
    end
    
    body = expr.args[3].args
    
    clean_fields = []
    schema_pairs = []
    field_names = []
    
    for line in body
        if line isa Expr && line.head == :(::)
            
            # CAS 1 : Double assertion -> b::Vector{T}::asSimplex(N)
            if line.args[1] isa Expr && line.args[1].head == :(::)
                inner_expr = line.args[1]
                field_name = inner_expr.args[1]
                field_type = inner_expr.args[2]
                constraint = line.args[2]
                
                push!(field_names, field_name)
                push!(clean_fields, :($field_name::$field_type))
                push!(schema_pairs, :($field_name = $constraint))
                
            # CAS 2 : Une seule assertion -> x::MyStruct{T, N} (Imbrication automatique)
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
    
    # Génération du code avec injection des paramètres de type dans la méthode du schéma
    return esc(quote
        struct $struct_sig
            $(clean_fields...)
            
        end
        
        # Capture DYNAMIQUE des paramètres de type (ex: T, N) pour le schéma
        function Paramorph.transformation_schema(::Type{S}) where {$(type_params...), S<:$struct_name{$(type_params...)}}
            return as($schema_namedtuple)
        end

        function Paramorph.reconstruct_struct(::Type{S}, nt::NamedTuple) where {$(type_params...), S<:$struct_name{$(type_params...)}}
            args = map(fieldnames(S)) do field
                Paramorph.reconstruct_field(fieldtype(S, field), getproperty(nt, field))
            end
            return S(args...)
        end
    end)
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

# --- API Publique ---

dimension_intrinsique(T::Type) = dimension(transformation_schema(T))

function constraint(T::Type, x::Vector{<:Real})
    schema = transformation_schema(T)
    @assert length(x) == dimension(schema) "Taille incorrecte du vecteur."
    return reconstruct_struct(T, transform(schema, x))
end

function constraint_with_logjac(T::Type, x::Vector{<:Real})
    schema = transformation_schema(T)
    @assert length(x) == dimension(schema) "Taille incorrecte du vecteur."
    nt, logjac = transform_and_logjac(schema, x)
    return reconstruct_struct(T, nt), logjac
end

function unconstrain(obj)
    schema = transformation_schema(typeof(obj))
    return inverse(schema, to_named_tuple(obj))
end

end # module
