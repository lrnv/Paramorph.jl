from pathlib import Path

p = Path('src/Paramorph.jl')
text = p.read_text()
old = '''    inferred_constructor = nothing
    if numeric_index == length(type_args)
'''
new = '''    inferred_constructor = nothing
    # Runtime auxiliary fields usually carry domain-level construction semantics
    # (dimensions, topology, labels, etc.).  Generating an unparameterized outer
    # constructor for those structs can conflict with intentional package
    # constructors whose signatures interpret the auxiliary arguments.  Keep
    # Paramorph responsible only for the fully-parameterized validating
    # constructor in that case.
    if numeric_index == length(type_args) && isempty(auxiliary_names)
'''
assert old in text
p.write_text(text.replace(old, new, 1))

p = Path('test/runtests.jl')
text = p.read_text()
repls = {
    'RuntimeSizedSimplex(3, [0.2, 0.3, 0.5])': 'RuntimeSizedSimplex{Float64}(3, [0.2, 0.3, 0.5])',
    'OpaqueVectorGeometry(3, [0.2, -0.4, 0.8])': 'OpaqueVectorGeometry{Float64}(3, [0.2, -0.4, 0.8])',
    '''RuntimeCompositeGeometry(\n            2,''': '''RuntimeCompositeGeometry{Float64}(\n            2,''',
    'PositiveScalarParameter(2.0, "custom")': 'PositiveScalarParameter{Float64}(2.0, "custom")',
}
for old, new in repls.items():
    assert old in text, old
    text = text.replace(old, new)
p.write_text(text)

p = Path('docs/src/tutorial.md')
text = p.read_text()
marker = 'Fields without `~` are auxiliary'
if marker in text and 'fully parameterized constructor' not in text:
    text = text.replace(marker, marker + '. Structs with auxiliary fields intentionally do not receive an inferred unparameterized outer constructor; use the fully parameterized constructor or define a domain-specific outer constructor')
p.write_text(text)
