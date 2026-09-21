from pathlib import Path

path = Path("src/Paramorph.jl")
text = path.read_text()

# Generated schema methods execute in the caller module, so Paramorph-owned
# TransformVariables references must remain qualified after macro expansion.
text = text.replace(
    'value_schema_expr = :(TransformVariables.as(($(value_schema_pairs...),)))',
    'value_schema_expr = :(Paramorph.TransformVariables.as(($(value_schema_pairs...),)))',
)
text = text.replace(
    'type_schema_expr = :(TransformVariables.as(($(type_schema_pairs...),)))',
    'type_schema_expr = :(Paramorph.TransformVariables.as(($(type_schema_pairs...),)))',
)

old = '''    struct_body = Expr(:block, clean_fields...)
    struct_expr = supertype === nothing ?
        Expr(:struct, false, declaration, struct_body) :
        Expr(:struct, false, Expr(:(<:), declaration, supertype), struct_body)

    methods = quote
'''
new = '''    # `new` is only legal in an inner constructor.  Reconstruction uses a
    # private trusted token because the transform has already established that
    # the constrained value belongs to the declared geometry.
    trusted_constructor = quote
        function $struct_name{$(type_args...)}(
            ::Paramorph.TrustedConstruction,
            $(clean_fields...),
        ) where {$(type_params...)}
            return new{$(type_args...)}($(r.name for r in field_records...))
        end
    end
    struct_body = Expr(:block, clean_fields..., trusted_constructor.args...)
    struct_expr = supertype === nothing ?
        Expr(:struct, false, declaration, struct_body) :
        Expr(:struct, false, Expr(:(<:), declaration, supertype), struct_body)

    # Julia's default outer constructors would bypass geometry validation.  For
    # the common layout where the numeric parameter is final, also recreate the
    # convenient partially-parameterized constructor and infer the numeric type
    # from the typed field arguments.
    inferred_constructor = nothing
    if numeric_index == length(type_args)
        prefix_args = type_args[1:end-1]
        prefix_params = type_params[1:end-1]
        target = isempty(prefix_args) ? struct_name : :($struct_name{$(prefix_args...)})
        inferred_constructor = quote
            function $target($(clean_fields...)) where {$(prefix_params...), $numeric_parameter_decl}
                values = $all_values
                Paramorph._validate_constrained(
                    $struct_name{$(type_args...)}, values,
                )
                return $struct_name{$(type_args...)}(
                    Paramorph._trusted_construction,
                    $(r.name for r in field_records...),
                )
            end
        end
    end

    methods = quote
'''
assert old in text, "struct constructor block changed unexpectedly"
text = text.replace(old, new, 1)

old = '''        function $struct_name{$(type_args...)}(
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
'''
new = '''        function $struct_name{$(type_args...)}($(clean_fields...)) where {$(type_params...)}
            values = $all_values
            Paramorph._validate_constrained($struct_name{$(type_args...)}, values)
            return $struct_name{$(type_args...)}(
                Paramorph._trusted_construction,
                $(r.name for r in field_records...),
            )
        end

        $inferred_constructor
'''
assert old in text, "outer constructor block changed unexpectedly"
text = text.replace(old, new, 1)

path.write_text(text)
print("macro hygiene and trusted-constructor patch applied")
