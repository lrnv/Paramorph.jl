from pathlib import Path

path = Path("src/Paramorph.jl")
text = path.read_text()

needle = '''    bind_from_values = [:( $(r.name) = getproperty(values, $(QuoteNode(r.name))) ) for r in field_records]
'''
replacement = needle + '''    bind_from_prototype = [
        :( $(r.name) = getproperty(prototype, $(QuoteNode(r.name))) )
        for r in field_records
    ]
'''
assert needle in text, "field binding block changed"
text = text.replace(needle, replacement, 1)

needle = '''        function Paramorph._reconstruct_declared_from_prototype(
            prototype::$struct_name, ::Type{S}, constrained::NamedTuple, context::NamedTuple,
        ) where {$(type_params...), S<:$struct_name{$(type_args...)}}
            return S(Paramorph._trusted_construction, $(prototype_field_values...))
        end
'''
replacement = '''        function Paramorph._reconstruct_declared_from_prototype(
            prototype::$struct_name, ::Type{S}, constrained::NamedTuple, context::NamedTuple,
        ) where {$(type_params...), S<:$struct_name{$(type_args...)}}
            $(bind_from_prototype...)
            return S(Paramorph._trusted_construction, $(prototype_field_values...))
        end
'''
assert needle in text, "prototype reconstruction block changed"
text = text.replace(needle, replacement, 1)

path.write_text(text)
print("prototype reconstruction now binds declared fields")
