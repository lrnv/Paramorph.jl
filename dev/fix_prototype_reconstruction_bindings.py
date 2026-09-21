from pathlib import Path

path = Path("src/Paramorph.jl")
text = path.read_text()

needle = '''    inferred_constructor = nothing
    if numeric_index == length(type_args)
        prefix_args = type_args[1:end-1]
        prefix_params = type_params[1:end-1]
        target = isempty(prefix_args) ? struct_name : :($struct_name{$(prefix_args...)})
        inferred_constructor = quote
            function $target($(clean_fields...)) where {$(prefix_params...), $numeric_parameter_decl}
'''
replacement = '''    inferred_constructor = nothing
    if numeric_index == length(type_args)
        prefix_args = type_args[1:end-1]
        prefix_params = type_params[1:end-1]
        target = isempty(prefix_args) ? struct_name : :($struct_name{$(prefix_args...)})
        # Auxiliary fields do not participate in numeric-type inference. Leaving
        # them untyped makes this convenience constructor less specific than
        # domain constructors that intentionally interpret auxiliary arguments.
        inferred_fields = [
            r.is_parameter ? :( $(r.name)::$(r.storage) ) : r.name
            for r in field_records
        ]
        inferred_constructor = quote
            function $target($(inferred_fields...)) where {$(prefix_params...), $numeric_parameter_decl}
'''
assert needle in text, "inferred constructor block changed"
text = text.replace(needle, replacement, 1)

path.write_text(text)
print("generated convenience constructor no longer specializes auxiliary fields")
