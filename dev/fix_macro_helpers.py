from pathlib import Path

path = Path("src/Paramorph.jl")
text = path.read_text()

old = '''_recursive_parameter_value(::NestedSpec, value, context) =
    parameter_values(value; context)
_recursive_parameter_value(spec::RecursiveSpec, value, context) =
    Tuple(parameter_values(child; context=spec.context) for child in value)
'''
new = '''_recursive_parameter_value(spec::NestedSpec, value, context) =
    parameter_values(value; context=spec.context)
_recursive_parameter_value(spec::RecursiveSpec, value, context) =
    Tuple(parameter_values(child; context=spec.context) for child in value)

# Ordinary transforms store their constrained value directly. Nested/recursive
# specifications instead expose the child's logical parameter value and rebuild
# the child object during reconstruction.
_parameter_value(::Any, value, context) = value
_parameter_value(spec::Union{NestedSpec,RecursiveSpec}, value, context) =
    _recursive_parameter_value(spec, value, context)

_reconstruct_parameter(::Any, ::Type, constrained, prototype, have_prototype::Bool) =
    constrained
function _reconstruct_parameter(
    spec::Union{NestedSpec,RecursiveSpec}, ::Type{T}, constrained,
    prototype, have_prototype::Bool,
) where {T}
    return _recursive_constraint(spec, T, constrained, prototype, have_prototype)
end
'''
assert old in text, "recursive parameter helper block changed"
text = text.replace(old, new, 1)

old_expr = '$(r.name for r in field_records...)'
new_expr = '$(map(r -> r.name, field_records)...)'
count = text.count(old_expr)
assert count == 3, f"expected three generated constructor splats, got {count}"
text = text.replace(old_expr, new_expr)

path.write_text(text)
print("Paramorph macro helper fixes applied")
