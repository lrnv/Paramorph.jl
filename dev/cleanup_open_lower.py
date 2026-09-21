from pathlib import Path

p = Path('src/transforms.jl')
text = p.read_text()
block = '''# Open lower bounds share the same forward map but reject the boundary in the
# inverse.  This distinction matters for constructor validation even though the
# optimizer chart only reaches the interior at finite coordinates.
struct OpenLower{L} <: TransformVariables.ScalarTransform
    lower::L
end

"""`open_lower(lower)` creates a strictly lower-bounded transform `(lower, ∞)`."""
open_lower(lower) = OpenLower(lower)
TransformVariables.transform(t::OpenLower, x::Number) = t.lower + exp(x)
TransformVariables.transform_and_logjac(t::OpenLower, x::Number) =
    (TransformVariables.transform(t, x), x)
function TransformVariables.inverse(t::OpenLower, y::Number)
    y > t.lower || throw(DomainError(y, "value must be greater than $(t.lower)"))
    return log(y - t.lower)
end
TransformVariables.inverse_eltype(::OpenLower, ::Type{T}) where {T<:Number} = float(T)

'''
count = text.count(block)
assert count >= 2, f'expected duplicate OpenLower block, got {count}'
first = text.index(block)
second = text.index(block, first + len(block))
text = text[:second] + text[second + len(block):]
p.write_text(text)

p = Path('src/Paramorph.jl')
text = p.read_text()
while text.count('    open_lower,\n') > 1:
    pos = text.rfind('    open_lower,\n')
    text = text[:pos] + text[pos + len('    open_lower,\n'):]
p.write_text(text)

p = Path('test/runtests.jl')
text = p.read_text()
struct_block = '''@paramorph T struct StrictPositiveParameter{T<:Real}
    x::T ~ open_lower(zero(T))
end
'''
while text.count(struct_block) > 1:
    pos = text.rfind(struct_block)
    text = text[:pos] + text[pos + len(struct_block):]
for line in (
    '        @test StrictPositiveParameter(1.0).x == 1.0\n',
    '        @test_throws DomainError StrictPositiveParameter(0.0)\n',
):
    while text.count(line) > 1:
        pos = text.rfind(line)
        text = text[:pos] + text[pos + len(line):]
p.write_text(text)
