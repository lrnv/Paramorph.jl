from pathlib import Path

p = Path('src/Paramorph.jl')
text = p.read_text()
text = text.replace('    closed_lower,\n    nonnegative,', '    closed_lower,\n    open_lower,\n    nonnegative,', 1)
p.write_text(text)

p = Path('src/transforms.jl')
text = p.read_text()
needle = '''"""`nonnegative()` creates a transform to the closed nonnegative half-line."""
nonnegative() = ClosedLower(0)
'''
insert = '''"""`nonnegative()` creates a transform to the closed nonnegative half-line."""
nonnegative() = ClosedLower(0)

# Open lower bounds share the same forward map but reject the boundary in the
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
assert needle in text
p.write_text(text.replace(needle, insert, 1))

p = Path('test/runtests.jl')
text = p.read_text()
needle = '''@paramorph T struct UnitIntervalParameter{T<:Real}
    p::T ~ bounded_interval(zero(T), one(T))
end
'''
insert = needle + '''\n@paramorph T struct StrictPositiveParameter{T<:Real}\n    x::T ~ open_lower(zero(T))\nend\n'''
assert needle in text
text = text.replace(needle, insert, 1)
needle = '''        @test_throws DomainError UnitIntervalParameter(1.1)
'''
insert = needle + '''        @test StrictPositiveParameter(1.0).x == 1.0\n        @test_throws DomainError StrictPositiveParameter(0.0)\n'''
assert needle in text
p.write_text(text.replace(needle, insert, 1))

p = Path('docs/src/index.md')
text = p.read_text()
if 'closed_lower' in text and 'open_lower' not in text:
    text = text.replace('closed_lower', 'closed_lower\nopen_lower', 1)
p.write_text(text)
