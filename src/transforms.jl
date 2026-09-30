# Internal composition used by `@paramorph` when a field geometry depends on
# parameter fields declared before it. Each builder receives the constrained
# values already produced during the current traversal.
struct ConditionalFieldTransform{N,B,R} <: TransformVariables.VectorTransform
    names::N
    builders::B
    reference::R
end

function _conditional_field_transform(names, builders, reference)
    return ConditionalFieldTransform(names, builders, reference)
end

_conditional_singleton(name::Symbol, value) = NamedTuple{(name,)}((value,))

function _conditional_reference_walk(transform::ConditionalFieldTransform, operation)
    partial = NamedTuple()
    result = operation === :dimension ? 0 : Bool
    for (name, builder) in zip(transform.names, transform.builders)
        local_transform = builder(partial)
        value = getproperty(transform.reference, name)
        if operation === :dimension
            result += TransformVariables.dimension(local_transform)
        else
            result = promote_type(
                result,
                TransformVariables.inverse_eltype(local_transform, typeof(value)),
            )
        end
        partial = merge(partial, _conditional_singleton(name, value))
    end
    return result
end

TransformVariables.dimension(transform::ConditionalFieldTransform) =
    _conditional_reference_walk(transform, :dimension)
TransformVariables.inverse_eltype(transform::ConditionalFieldTransform, ::Type) =
    _conditional_reference_walk(transform, :inverse_eltype)

function TransformVariables.transform_with(
    flag::TransformVariables.LogJacFlag,
    transform::ConditionalFieldTransform,
    coordinates::AbstractVector,
    index,
)
    values = NamedTuple()
    logjac = TransformVariables.logjac_zero(flag, eltype(coordinates))
    for (name, builder) in zip(transform.names, transform.builders)
        local_transform = builder(values)
        value, contribution, index = TransformVariables.transform_with(
            flag, local_transform, coordinates, index,
        )
        values = merge(values, _conditional_singleton(name, value))
        logjac += contribution
    end
    return values, logjac, index
end

function TransformVariables.inverse_at!(
    coordinates::AbstractVector,
    index,
    transform::ConditionalFieldTransform,
    values::NamedTuple,
)
    partial = NamedTuple()
    for (name, builder) in zip(transform.names, transform.builders)
        local_transform = builder(partial)
        value = getproperty(values, name)
        index = TransformVariables.inverse_at!(
            coordinates, index, local_transform, value,
        )
        partial = merge(partial, _conditional_singleton(name, value))
    end
    return index
end

# Implementation type for `closed_lower`.
struct ClosedLower{L} <: TransformVariables.ScalarTransform
    lower::L
end

"""`closed_lower(lower)` creates a lower-bounded transform whose inverse accepts the boundary."""
closed_lower(lower) = ClosedLower(lower)
"""`nonnegative()` creates a transform to the closed nonnegative half-line."""
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

TransformVariables.transform(t::ClosedLower, x::Number) = t.lower + exp(x)
TransformVariables.transform_and_logjac(t::ClosedLower, x::Number) =
    (TransformVariables.transform(t, x), x)
function TransformVariables.inverse(t::ClosedLower, y::Number)
    y >= t.lower || throw(DomainError(y, "value must be at least $(t.lower)"))
    return log(y - t.lower)
end
TransformVariables.inverse_eltype(::ClosedLower, ::Type{T}) where {T<:Number} = float(T)

# Implementation type for `bounded_interval`.
struct BoundedInterval{L,U} <: TransformVariables.ScalarTransform
    lower::L
    upper::U
    left_closed::Bool
    right_closed::Bool
    function BoundedInterval(lower, upper, left_closed, right_closed)
        lower < upper || throw(ArgumentError("lower bound must be smaller than upper bound"))
        new{typeof(lower),typeof(upper)}(lower, upper, left_closed, right_closed)
    end
end

"""`bounded_interval(lower, upper; left_closed=true, right_closed=true)` creates an interval transform."""
bounded_interval(lower, upper; left_closed=true, right_closed=true) =
    BoundedInterval(lower, upper, left_closed, right_closed)

function TransformVariables.transform(t::BoundedInterval, x::Number)
    q = inv(one(x) + exp(-x))
    return t.lower + (t.upper - t.lower) * q
end
function TransformVariables.transform_and_logjac(t::BoundedInterval, x::Number)
    y = TransformVariables.transform(t, x)
    q = (y - t.lower) / (t.upper - t.lower)
    return y, log(t.upper - t.lower) + log(q) + log1p(-q)
end
function TransformVariables.inverse(t::BoundedInterval, y::Number)
    left_ok = t.left_closed ? y >= t.lower : y > t.lower
    right_ok = t.right_closed ? y <= t.upper : y < t.upper
    left_ok && right_ok || throw(DomainError(y, "value is outside the configured interval"))
    q = (y - t.lower) / (t.upper - t.lower)
    return log(q) - log1p(-q)
end
TransformVariables.inverse_eltype(::BoundedInterval, ::Type{T}) where {T<:Number} = float(T)

# Implementation type for `correlation_matrix`.
struct CorrelationMatrix <: TransformVariables.VectorTransform
    n::Int
    factor::TransformVariables.CorrCholeskyFactor
    function CorrelationMatrix(n::Integer)
        n > 0 || throw(ArgumentError("matrix dimension must be positive"))
        new(Int(n), TransformVariables.corr_cholesky_factor(Int(n)))
    end
end

"""`correlation_matrix(n)` transforms coordinates into an `n × n` correlation matrix."""
correlation_matrix(n) = CorrelationMatrix(n)
TransformVariables.dimension(t::CorrelationMatrix) = TransformVariables.dimension(t.factor)

_correlation_value(t::CorrelationMatrix, x) = begin
    U = TransformVariables.transform(t.factor, x)
    Matrix(transpose(U) * U)
end

function TransformVariables.transform_with(flag::TransformVariables.NoLogJac,
        t::CorrelationMatrix, x::AbstractVector, index)
    n = TransformVariables.dimension(t)
    coordinates = @view x[index:(index + n - 1)]
    return _correlation_value(t, coordinates), flag, index + n
end

function TransformVariables.transform_with(flag::TransformVariables.LogJac,
        t::CorrelationMatrix, x::AbstractVector, index)
    n = TransformVariables.dimension(t)
    coordinates = @view x[index:(index + n - 1)]
    flatten(R) = [R[i, j] for j in 1:t.n for i in (j + 1):t.n]
    R, logjac = TransformVariables.value_and_logjac_forwarddiff(
        Base.Fix1(_correlation_value, t), coordinates; flatten,
    )
    return R, logjac, index + n
end

function TransformVariables.inverse_eltype(::CorrelationMatrix,
        ::Type{M}) where {T,M<:AbstractMatrix{T}}
    return float(T)
end

function TransformVariables.inverse_at!(x::AbstractVector, index,
        t::CorrelationMatrix, R::AbstractMatrix)
    size(R) == (t.n, t.n) || throw(DimensionMismatch("expected a $(t.n) × $(t.n) matrix"))
    isapprox(R, transpose(R)) || throw(DomainError(R, "correlation matrix must be symmetric"))
    all(i -> isapprox(R[i, i], one(R[i, i])), 1:t.n) ||
        throw(DomainError(R, "correlation matrix must have a unit diagonal"))
    U = cholesky(Symmetric(R)).U
    return TransformVariables.inverse_at!(x, index, t.factor, U)
end

# General symmetric positive-definite matrices. Coordinates are the lower
# triangle of a Cholesky factor; diagonal coordinates are exponentiated.
struct PositiveDefiniteMatrix <: TransformVariables.VectorTransform
    n::Int
    function PositiveDefiniteMatrix(n::Integer)
        n > 0 || throw(ArgumentError("matrix dimension must be positive"))
        new(Int(n))
    end
end

"""`positive_definite_matrix(n)` transforms coordinates into an `n × n` SPD matrix."""
positive_definite_matrix(n) = PositiveDefiniteMatrix(n)
TransformVariables.dimension(t::PositiveDefiniteMatrix) = t.n * (t.n + 1) ÷ 2

function _positive_definite_value(t::PositiveDefiniteMatrix, x)
    T = eltype(x)
    L = zeros(T, t.n, t.n)
    k = firstindex(x)
    for j in 1:t.n, i in j:t.n
        L[i, j] = i == j ? exp(x[k]) : x[k]
        k += 1
    end
    return L * transpose(L)
end

function TransformVariables.transform_with(flag::TransformVariables.NoLogJac,
        t::PositiveDefiniteMatrix, x::AbstractVector, index)
    n = TransformVariables.dimension(t)
    value = _positive_definite_value(t, @view x[index:(index + n - 1)])
    return value, flag, index + n
end
function TransformVariables.transform_with(::TransformVariables.LogJac,
        t::PositiveDefiniteMatrix, x::AbstractVector, index)
    n = TransformVariables.dimension(t)
    coordinates = @view x[index:(index + n - 1)]
    flatten(S) = [S[i, j] for j in 1:t.n for i in j:t.n]
    value, logjac = TransformVariables.value_and_logjac_forwarddiff(
        Base.Fix1(_positive_definite_value, t), coordinates; flatten,
    )
    return value, logjac, index + n
end
TransformVariables.inverse_eltype(::PositiveDefiniteMatrix,
    ::Type{M}) where {T,M<:AbstractMatrix{T}} = float(T)

function TransformVariables.inverse_at!(x::AbstractVector, index,
        t::PositiveDefiniteMatrix, S::AbstractMatrix)
    size(S) == (t.n, t.n) || throw(DimensionMismatch("expected a $(t.n) × $(t.n) matrix"))
    isapprox(S, transpose(S)) || throw(DomainError(S, "matrix must be symmetric"))
    L = try
        cholesky(Symmetric(S)).L
    catch error
        error isa LinearAlgebra.PosDefException || rethrow()
        throw(DomainError(S, "matrix must be positive definite"))
    end
    for j in 1:t.n, i in j:t.n
        x[index] = i == j ? log(L[i, j]) : L[i, j]
        index += 1
    end
    return index
end

# Strict Hüsler--Reiss variograms are linear images of SPD Gram matrices on
# d-1 anchored points.
struct VariogramMatrix <: TransformVariables.VectorTransform
    d::Int
    gram::PositiveDefiniteMatrix
    function VariogramMatrix(d::Integer)
        d >= 2 || throw(ArgumentError("variogram dimension must be at least two"))
        new(Int(d), PositiveDefiniteMatrix(Int(d) - 1))
    end
end

"""`variogram_matrix(d)` transforms coordinates into a strict `d × d` variogram."""
variogram_matrix(d) = VariogramMatrix(d)
TransformVariables.dimension(t::VariogramMatrix) = TransformVariables.dimension(t.gram)

function _gram_to_variogram(S)
    n = size(S, 1)
    T = eltype(S)
    Γ = zeros(T, n + 1, n + 1)
    for j in 1:n, i in 1:n
        Γ[i, j] = S[i, i] + S[j, j] - 2S[i, j]
    end
    for i in 1:n
        Γ[i, n + 1] = Γ[n + 1, i] = S[i, i]
    end
    return Γ
end

function _variogram_to_gram(t::VariogramMatrix, Γ)
    size(Γ) == (t.d, t.d) || throw(DimensionMismatch("expected a $(t.d) × $(t.d) matrix"))
    isapprox(Γ, transpose(Γ)) || throw(DomainError(Γ, "variogram must be symmetric"))
    all(i -> isapprox(Γ[i, i], zero(Γ[i, i])), 1:t.d) ||
        throw(DomainError(Γ, "variogram must have a zero diagonal"))
    n = t.d - 1
    return [((Γ[i, t.d] + Γ[j, t.d] - Γ[i, j]) / 2) for i in 1:n, j in 1:n]
end

_variogram_value(t::VariogramMatrix, x) =
    _gram_to_variogram(_positive_definite_value(t.gram, x))

function TransformVariables.transform_with(flag::TransformVariables.NoLogJac,
        t::VariogramMatrix, x::AbstractVector, index)
    n = TransformVariables.dimension(t)
    value = _variogram_value(t, @view x[index:(index + n - 1)])
    return value, flag, index + n
end
function TransformVariables.transform_with(::TransformVariables.LogJac,
        t::VariogramMatrix, x::AbstractVector, index)
    n = TransformVariables.dimension(t)
    coordinates = @view x[index:(index + n - 1)]
    flatten(Γ) = [Γ[i, j] for j in 1:t.d for i in (j + 1):t.d]
    value, logjac = TransformVariables.value_and_logjac_forwarddiff(
        Base.Fix1(_variogram_value, t), coordinates; flatten,
    )
    return value, logjac, index + n
end
TransformVariables.inverse_eltype(::VariogramMatrix,
    ::Type{M}) where {T,M<:AbstractMatrix{T}} = float(T)
function TransformVariables.inverse_at!(x::AbstractVector, index,
        t::VariogramMatrix, Γ::AbstractMatrix)
    S = _variogram_to_gram(t, Γ)
    return TransformVariables.inverse_at!(x, index, t.gram, S)
end

# A d-vector in the strict positive orthant with a strict upper bound on its
# sum, represented by the first d entries of a scaled (d+1)-simplex.
struct PositiveVectorWithSumBelow{L} <: TransformVariables.VectorTransform
    limit::L
    d::Int
    simplex::TransformVariables.UnitSimplex
end

function positive_vector_with_sum_below(limit, d::Integer)
    limit > zero(limit) || throw(ArgumentError("sum limit must be positive"))
    d > 0 || throw(ArgumentError("vector dimension must be positive"))
    return PositiveVectorWithSumBelow(limit, Int(d), TransformVariables.UnitSimplex(Int(d) + 1))
end
TransformVariables.dimension(t::PositiveVectorWithSumBelow) = t.d

function _sum_bounded_value(t::PositiveVectorWithSumBelow, x)
    simplex = TransformVariables.transform(t.simplex, x)
    return t.limit .* simplex[1:t.d]
end
function TransformVariables.transform_with(flag::TransformVariables.NoLogJac,
        t::PositiveVectorWithSumBelow, x::AbstractVector, index)
    value = _sum_bounded_value(t, @view x[index:(index + t.d - 1)])
    return value, flag, index + t.d
end
function TransformVariables.transform_with(::TransformVariables.LogJac,
        t::PositiveVectorWithSumBelow, x::AbstractVector, index)
    coordinates = @view x[index:(index + t.d - 1)]
    value, logjac = TransformVariables.value_and_logjac_forwarddiff(
        Base.Fix1(_sum_bounded_value, t), coordinates,
    )
    return value, logjac, index + t.d
end
TransformVariables.inverse_eltype(::PositiveVectorWithSumBelow,
    ::Type{V}) where {T,V<:AbstractVector{T}} = float(T)
function TransformVariables.inverse_at!(x::AbstractVector, index,
        t::PositiveVectorWithSumBelow, values::AbstractVector)
    length(values) == t.d || throw(DimensionMismatch("expected $(t.d) values"))
    all(>=(zero(eltype(values))), values) || throw(DomainError(values, "values must be nonnegative"))
    slack = t.limit - sum(values)
    slack >= zero(slack) || throw(DomainError(values, "values must sum to at most $(t.limit)"))
    simplex = [values ./ t.limit; slack / t.limit]
    return TransformVariables.inverse_at!(x, index, t.simplex, simplex)
end

# Smooth square-to-quadrilateral chart used by the asymmetric Mixed family.
struct AsymmetricMixed <: TransformVariables.VectorTransform
    box::TransformVariables.TransformTuple
end
asymmetric_mixed() = AsymmetricMixed(TransformVariables.as((u=TransformVariables.as𝕀, v=TransformVariables.as𝕀)))
TransformVariables.dimension(::AsymmetricMixed) = 2

function _asymmetric_mixed_value(t::AsymmetricMixed, x)
    p = TransformVariables.transform(t.box, x)
    return (; θ₁=p.u * (3 - p.v) / 2, θ₂=(p.v - p.u) / 2)
end
function TransformVariables.transform_with(flag::TransformVariables.NoLogJac,
        t::AsymmetricMixed, x::AbstractVector, index)
    value = _asymmetric_mixed_value(t, @view x[index:(index + 1)])
    return value, flag, index + 2
end
function TransformVariables.transform_with(::TransformVariables.LogJac,
        t::AsymmetricMixed, x::AbstractVector, index)
    coordinates = @view x[index:(index + 1)]
    flatten(p) = [p.θ₁, p.θ₂]
    value, logjac = TransformVariables.value_and_logjac_forwarddiff(
        Base.Fix1(_asymmetric_mixed_value, t), coordinates; flatten,
    )
    return value, logjac, index + 2
end
function TransformVariables.inverse_eltype(::AsymmetricMixed,
        ::Type{NamedTuple{N,Tuple{T,T}}}) where {N,T}
    return float(T)
end
function TransformVariables.inverse_at!(x::AbstractVector, index,
        t::AsymmetricMixed, p::NamedTuple)
    hasproperty(p, :θ₁) && hasproperty(p, :θ₂) ||
        throw(ArgumentError("expected fields θ₁ and θ₂"))
    discriminant = (3 - 2p.θ₂)^2 - 8p.θ₁
    discriminant >= zero(discriminant) || throw(DomainError(p, "invalid asymmetric Mixed parameters"))
    u = ((3 - 2p.θ₂) - sqrt(discriminant)) / 2
    v = u + 2p.θ₂
    coordinates = TransformVariables.inverse(t.box, (; u, v))
    x[index:(index + 1)] .= coordinates
    return index + 2
end

# Implementation type for `repeat_transform`.
struct RepeatedTransform{S} <: TransformVariables.VectorTransform
    inner::S
    count::Int
    function RepeatedTransform(inner, count::Integer)
        count >= 0 || throw(ArgumentError("repeat count must be nonnegative"))
        new{typeof(inner)}(inner, Int(count))
    end
end

"""`repeat_transform(inner, count)` repeats a transformation and returns a vector of results."""
repeat_transform(inner, count) = RepeatedTransform(inner, count)
TransformVariables.dimension(t::RepeatedTransform) =
    t.count * TransformVariables.dimension(t.inner)

function TransformVariables.transform_with(flag::TransformVariables.LogJacFlag,
        t::RepeatedTransform, x::AbstractVector, index)
    output = Vector{Any}(undef, t.count)
    logjac = TransformVariables.logjac_zero(flag, eltype(x))
    for i in eachindex(output)
        output[i], contribution, index =
            TransformVariables.transform_with(flag, t.inner, x, index)
        logjac += contribution
    end
    isempty(output) && return output, logjac, index
    T = promote_type(map(typeof, output)...)
    return T[output...], logjac, index
end

function TransformVariables.inverse_eltype(t::RepeatedTransform,
        ::Type{V}) where {E,V<:AbstractVector{E}}
    return TransformVariables.inverse_eltype(t.inner, E)
end

function TransformVariables.inverse_at!(x::AbstractVector, index,
        t::RepeatedTransform, values::AbstractVector)
    length(values) == t.count || throw(DimensionMismatch("incorrect repeated value count"))
    for value in values
        index = TransformVariables.inverse_at!(x, index, t.inner, value)
    end
    return index
end

# Implementation type for `joint_transform`.
struct JointTransform{G,F,B,H} <: TransformVariables.VectorTransform
    base::G
    forward::F
    backward::B
    flatten::H
end

"""`joint_transform(base, forward, backward; flatten=identity)` defines a coupled bijection."""
joint_transform(base, forward, backward; flatten=identity) =
    JointTransform(base, forward, backward, flatten)
TransformVariables.dimension(t::JointTransform) = TransformVariables.dimension(t.base)

function TransformVariables.transform_with(flag::TransformVariables.NoLogJac,
        t::JointTransform, x::AbstractVector, index)
    value, _, next = TransformVariables.transform_with(flag, t.base, x, index)
    return t.forward(value), flag, next
end
function TransformVariables.transform_with(::TransformVariables.LogJac,
        t::JointTransform, x::AbstractVector, index)
    n = TransformVariables.dimension(t)
    coordinates = @view x[index:(index + n - 1)]
    f(z) = t.forward(TransformVariables.transform(t.base, z))
    value, logjac = TransformVariables.value_and_logjac_forwarddiff(
        f, coordinates; flatten=t.flatten,
    )
    return value, logjac, index + n
end
function TransformVariables.inverse(t::JointTransform, value)
    return TransformVariables.inverse(t.base, t.backward(value))
end
TransformVariables.inverse_eltype(::JointTransform,
    ::Type{V}) where {T,V<:AbstractVector{T}} = float(T)
function TransformVariables.inverse_at!(x::AbstractVector, index,
        t::JointTransform, value)
    coordinates = TransformVariables.inverse(t, value)
    x[index:(index + length(coordinates) - 1)] .= coordinates
    return index + length(coordinates)
end

# Zero-dimensional transform used when a prototype-driven recursive collection
# contains an opaque child. The child is structural state, not an optimizer
# parameter, and is reproduced exactly.
struct FixedValueTransform{V} <: TransformVariables.VectorTransform
    value::V
end
TransformVariables.dimension(::FixedValueTransform) = 0
function TransformVariables.transform_with(
    flag::TransformVariables.LogJacFlag,
    transform::FixedValueTransform,
    x::AbstractVector,
    index,
)
    return transform.value,
           TransformVariables.logjac_zero(flag, eltype(x)), index
end
TransformVariables.inverse_eltype(::FixedValueTransform, ::Type) = Bool
function TransformVariables.inverse_at!(
    x::AbstractVector, index, transform::FixedValueTransform, value,
)
    isequal(value, transform.value) || throw(DomainError(
        value,
        "fixed prototype value cannot be changed",
    ))
    return index
end

"""
    simplex_face(mask)

Transform Euclidean coordinates to the face of a simplex selected by Boolean
`mask`. Active entries are positive and sum to one; inactive entries are
structural zeros. The inverse rejects values outside that face.
"""
function simplex_face(mask)
    active_mask = Bool[mask...]
    active = findall(active_mask)
    isempty(active) && throw(ArgumentError("a simplex face needs at least one active entry"))
    inactive = findall(!, active_mask)
    base = TransformVariables.UnitSimplex(length(active))
    forward = function(values)
        result = zeros(eltype(values), length(active_mask))
        result[active] .= values
        return result
    end
    backward = function(values)
        length(values) == length(active_mask) || throw(DimensionMismatch(
            "expected $(length(active_mask)) simplex entries",
        ))
        all(i -> iszero(values[i]), inactive) || throw(DomainError(
            values,
            "inactive simplex-face entries must be zero",
        ))
        return collect(values[active])
    end
    return joint_transform(base, forward, backward)
end

"""
    recursive_tree(prototype; node_transform, node_value, children, rebuild)

Build a prototype-driven transformation for a runtime tree. `node_transform`
is called as `(node, parent_value, index)` and returns the local transform, or
`nothing` for a fixed node. `node_value` extracts the local constrained value,
`children` returns its ordered children, and `rebuild` constructs a node as
`rebuild(prototype_node, new_value, new_children)`.

During forward transformation, child transforms receive the newly reconstructed
parent value. A four-argument callback `(node, parent_value, index, numeric_type)`
may use the coordinate element type explicitly; three-argument callbacks remain
supported. The topology comes from `prototype` and must remain unchanged.
Local transform dimensions must not change when their parent value changes.
"""
struct RecursiveTreeTransform{P,F,V,C,R} <: TransformVariables.VectorTransform
    prototype::P
    node_transform::F
    node_value::V
    children::C
    rebuild::R
end

"""
    recursive_tree(prototype; node_transform, node_value, children, rebuild)

Construct a prototype-driven recursive tree transformation whose child schemas
may depend on newly reconstructed parent values.
"""
recursive_tree(prototype; node_transform, node_value, children, rebuild) =
    RecursiveTreeTransform(prototype, node_transform, node_value, children, rebuild)

function _tree_node_transform(transform, node, parent, index::Int, ::Type{N}) where {N}
    if applicable(transform.node_transform, node, parent, index, N)
        return transform.node_transform(node, parent, index, N)
    end
    return transform.node_transform(node, parent, index)
end

function _tree_dimension(transform::RecursiveTreeTransform, node, parent, index::Int)
    value = transform.node_value(node)
    local_transform = _tree_node_transform(
        transform, node, parent, index, typeof(value),
    )
    dimension = local_transform === nothing ? 0 :
                TransformVariables.dimension(local_transform)
    parent_value = value
    for (i, child) in pairs(transform.children(node))
        dimension += _tree_dimension(transform, child, parent_value, Int(i))
    end
    return dimension
end

TransformVariables.dimension(transform::RecursiveTreeTransform) =
    _tree_dimension(transform, transform.prototype, nothing, 1)

function _transform_tree(
    flag, transform::RecursiveTreeTransform, prototype, parent_value,
    child_index::Int, coordinates, index,
)
    local_transform = _tree_node_transform(
        transform, prototype, parent_value, child_index, eltype(coordinates),
    )
    if local_transform === nothing
        value = transform.node_value(prototype)
        logjac = TransformVariables.logjac_zero(flag, eltype(coordinates))
    else
        value, logjac, index = TransformVariables.transform_with(
            flag, local_transform, coordinates, index,
        )
    end

    prototype_children = transform.children(prototype)
    rebuilt_children = map(eachindex(prototype_children)) do i
        child = prototype_children[i]
        rebuilt, contribution, index = _transform_tree(
            flag, transform, child, value, Int(i), coordinates, index,
        )
        logjac += contribution
        return rebuilt
    end
    return transform.rebuild(prototype, value, rebuilt_children), logjac, index
end

function TransformVariables.transform_with(
    flag::TransformVariables.LogJacFlag,
    transform::RecursiveTreeTransform,
    coordinates::AbstractVector,
    index,
)
    return _transform_tree(
        flag, transform, transform.prototype, nothing, 1, coordinates, index,
    )
end

function _tree_inverse_eltype(transform::RecursiveTreeTransform, node, parent, index::Int)
    value = transform.node_value(node)
    local_transform = _tree_node_transform(
        transform, node, parent, index, typeof(value),
    )
    T = local_transform === nothing ? Bool : TransformVariables.inverse_eltype(
        local_transform,
        typeof(value),
    )
    parent_value = value
    for (i, child) in pairs(transform.children(node))
        T = promote_type(T, _tree_inverse_eltype(
            transform, child, parent_value, Int(i),
        ))
    end
    return T
end

TransformVariables.inverse_eltype(transform::RecursiveTreeTransform, ::Type) =
    _tree_inverse_eltype(transform, transform.prototype, nothing, 1)

function _inverse_tree!(
    coordinates, index, transform::RecursiveTreeTransform,
    node, prototype_node, parent_value, child_index::Int,
)
    value = transform.node_value(node)
    local_transform = _tree_node_transform(
        transform, node, parent_value, child_index, typeof(value),
    )
    if local_transform !== nothing
        index = TransformVariables.inverse_at!(
            coordinates, index, local_transform, value,
        )
    end
    node_children = transform.children(node)
    prototype_children = transform.children(prototype_node)
    length(node_children) == length(prototype_children) || throw(DimensionMismatch(
        "recursive tree topology differs from its prototype",
    ))
    for (i, child) in pairs(node_children)
        index = _inverse_tree!(
            coordinates, index, transform, child, prototype_children[i], value, Int(i),
        )
    end
    return index
end

function TransformVariables.inverse_at!(
    coordinates::AbstractVector,
    index,
    transform::RecursiveTreeTransform,
    tree,
)
    return _inverse_tree!(
        coordinates, index, transform, tree, transform.prototype, nothing, 1,
    )
end

# Implementation type for `polytope`.
struct PolytopeTransform{MA,VB,VC} <: TransformVariables.VectorTransform
    A::MA
    b::VB
    center::VC
end

"""
    polytope(A, b; center=zeros(...))

Map Euclidean coordinates bijectively to the interior of the bounded polytope
`A * y <= b`, using radial projection from a strictly interior point.
The map is piecewise smooth where the active facet changes.
"""
function polytope(A::AbstractMatrix, b::AbstractVector; center=zeros(promote_type(eltype(A), eltype(b)), size(A, 2)))
    size(A, 1) == length(b) || throw(DimensionMismatch("A and b have incompatible sizes"))
    size(A, 2) == length(center) || throw(DimensionMismatch("center has the wrong dimension"))
    slack = b - A * center
    all(>(zero(eltype(slack))), slack) || throw(ArgumentError("center must lie strictly inside the polytope"))
    return PolytopeTransform(copy(A), copy(b), collect(center))
end

TransformVariables.dimension(t::PolytopeTransform) = length(t.center)

function _radial_limit(t::PolytopeTransform, direction)
    products = t.A * direction
    slack = t.b - t.A * t.center
    limit = oftype(first(slack), Inf)
    for i in eachindex(products)
        products[i] > zero(products[i]) && (limit = min(limit, slack[i] / products[i]))
    end
    isfinite(limit) || throw(ArgumentError("polytope must be bounded in every direction"))
    return limit
end

function _polytope_value(t::PolytopeTransform, x)
    radius = sqrt(sum(abs2, x))
    iszero(radius) && return t.center .+ zero(eltype(x))
    direction = x / radius
    ball_radius = radius / sqrt(one(radius) + radius^2)
    return t.center + (_radial_limit(t, direction) * ball_radius) * direction
end

function TransformVariables.transform_with(flag::TransformVariables.NoLogJac,
        t::PolytopeTransform, x::AbstractVector, index)
    n = TransformVariables.dimension(t)
    value = _polytope_value(t, @view x[index:(index + n - 1)])
    return value, flag, index + n
end
function TransformVariables.transform_with(::TransformVariables.LogJac,
        t::PolytopeTransform, x::AbstractVector, index)
    n = TransformVariables.dimension(t)
    coordinates = @view x[index:(index + n - 1)]
    value, logjac = TransformVariables.value_and_logjac_forwarddiff(
        Base.Fix1(_polytope_value, t), coordinates,
    )
    return value, logjac, index + n
end
TransformVariables.inverse_eltype(::PolytopeTransform,
    ::Type{V}) where {T,V<:AbstractVector{T}} = float(T)

function TransformVariables.inverse_at!(x::AbstractVector, index,
        t::PolytopeTransform, y::AbstractVector)
    length(y) == TransformVariables.dimension(t) || throw(DimensionMismatch("point has the wrong dimension"))
    all(t.A * y .<= t.b) || throw(DomainError(y, "point is outside the polytope"))
    delta = y - t.center
    distance = sqrt(sum(abs2, delta))
    if iszero(distance)
        x[index:(index + length(delta) - 1)] .= zero(eltype(x))
        return index + length(delta)
    end
    direction = delta / distance
    ball_radius = distance / _radial_limit(t, direction)
    ball_radius <= one(ball_radius) || throw(DomainError(y, "point is outside the polytope"))
    radius = ball_radius / sqrt((one(ball_radius) - ball_radius) * (one(ball_radius) + ball_radius))
    x[index:(index + length(delta) - 1)] .= radius .* direction
    return index + length(delta)
end

# Closed matrix geometries used by models whose natural constrained spaces have
# distinguished boundary points that finite Euclidean coordinates cannot reach.
export closed_correlation_matrix, compact_variogram_matrix

struct ClosedCorrelationMatrix{G} <: TransformVariables.VectorTransform
    n::Int
    interior::G
end

"""`closed_correlation_matrix(n)` transforms coordinates into an `n × n` correlation matrix, including boundaries."""
closed_correlation_matrix(n::Integer) =
    ClosedCorrelationMatrix(Int(n), correlation_matrix(n))
TransformVariables.dimension(t::ClosedCorrelationMatrix) =
    TransformVariables.dimension(t.interior)

_closed_complete_correlation(::Type{T}, n::Int) where {T} = ones(T, n, n)
_closed_is_complete_correlation(R::AbstractMatrix) = all(isone, R)

"""
    closed_correlation_matrix(n)

Extend `correlation_matrix(n)` with the all-ones correlation matrix as a closed
boundary point. That boundary is represented by an all-`+Inf` unconstrained
coordinate vector; its log-Jacobian is `-Inf`.
"""
function TransformVariables.transform_with(flag::TransformVariables.NoLogJac,
        t::ClosedCorrelationMatrix, x::AbstractVector, index)
    n = TransformVariables.dimension(t)
    coordinates = @view x[index:(index + n - 1)]
    if all(isinf, coordinates) && all(>(zero(eltype(coordinates))), coordinates)
        return _closed_complete_correlation(eltype(coordinates), t.n), flag, index + n
    end
    return TransformVariables.transform_with(flag, t.interior, x, index)
end
function TransformVariables.transform_with(::TransformVariables.LogJac,
        t::ClosedCorrelationMatrix, x::AbstractVector, index)
    n = TransformVariables.dimension(t)
    coordinates = @view x[index:(index + n - 1)]
    if all(isinf, coordinates) && all(>(zero(eltype(coordinates))), coordinates)
        return _closed_complete_correlation(eltype(coordinates), t.n), -Inf, index + n
    end
    return TransformVariables.transform_with(TransformVariables.LogJac(), t.interior, x, index)
end
TransformVariables.inverse_eltype(::ClosedCorrelationMatrix,
    ::Type{M}) where {T,M<:AbstractMatrix{T}} = float(T)
function TransformVariables.inverse_at!(x::AbstractVector, index,
        t::ClosedCorrelationMatrix, R::AbstractMatrix)
    size(R) == (t.n, t.n) || throw(DimensionMismatch("expected a $(t.n) × $(t.n) matrix"))
    all(isfinite, R) || throw(DomainError(R, "correlation matrix must contain only finite entries"))
    n = TransformVariables.dimension(t)
    if _closed_is_complete_correlation(R)
        fill!(@view(x[index:(index + n - 1)]), Inf)
        return index + n
    end
    try
        return TransformVariables.inverse_at!(x, index, t.interior, R)
    catch error
        error isa LinearAlgebra.PosDefException || rethrow()
        throw(DomainError(R, "correlation matrix must be positive definite or all ones"))
    end
end

struct CompactVariogramMatrix{G} <: TransformVariables.VectorTransform
    d::Int
    interior::G
end

"""`compact_variogram_matrix(d)` transforms coordinates into a strict `d × d` variogram, including boundaries."""
compact_variogram_matrix(d::Integer) =
    CompactVariogramMatrix(Int(d), variogram_matrix(d))
TransformVariables.dimension(t::CompactVariogramMatrix) =
    TransformVariables.dimension(t.interior)

_compact_zero_variogram(::Type{T}, d::Int) where {T} = zeros(T, d, d)
function _compact_independence_variogram(::Type{T}, d::Int) where {T}
    Γ = fill(T(Inf), d, d)
    @inbounds for i in 1:d
        Γ[i, i] = zero(T)
    end
    return Γ
end
_compact_is_zero_variogram(Γ::AbstractMatrix) = all(iszero, Γ)
function _compact_is_independence_variogram(Γ::AbstractMatrix)
    d1, d2 = size(Γ)
    d1 == d2 || return false
    @inbounds for j in 1:d1, i in 1:d1
        if i == j
            iszero(Γ[i, j]) || return false
        else
            isinf(Γ[i, j]) && Γ[i, j] > 0 || return false
        end
    end
    return true
end

"""
    compact_variogram_matrix(d)

Extend `variogram_matrix(d)` by adjoining the zero variogram and the extended
independence variogram with `+Inf` off-diagonal entries. They are represented by
all-`-Inf` and all-`+Inf` unconstrained coordinate vectors respectively; both
have log-Jacobian `-Inf`.
"""
function TransformVariables.transform_with(flag::TransformVariables.NoLogJac,
        t::CompactVariogramMatrix, x::AbstractVector, index)
    n = TransformVariables.dimension(t)
    coordinates = @view x[index:(index + n - 1)]
    if all(isinf, coordinates)
        if all(>(zero(eltype(coordinates))), coordinates)
            return _compact_independence_variogram(eltype(coordinates), t.d), flag, index + n
        elseif all(<(zero(eltype(coordinates))), coordinates)
            return _compact_zero_variogram(eltype(coordinates), t.d), flag, index + n
        end
    end
    return TransformVariables.transform_with(flag, t.interior, x, index)
end
function TransformVariables.transform_with(::TransformVariables.LogJac,
        t::CompactVariogramMatrix, x::AbstractVector, index)
    n = TransformVariables.dimension(t)
    coordinates = @view x[index:(index + n - 1)]
    if all(isinf, coordinates)
        if all(>(zero(eltype(coordinates))), coordinates)
            return _compact_independence_variogram(eltype(coordinates), t.d), -Inf, index + n
        elseif all(<(zero(eltype(coordinates))), coordinates)
            return _compact_zero_variogram(eltype(coordinates), t.d), -Inf, index + n
        end
    end
    return TransformVariables.transform_with(TransformVariables.LogJac(), t.interior, x, index)
end
TransformVariables.inverse_eltype(::CompactVariogramMatrix,
    ::Type{M}) where {T,M<:AbstractMatrix{T}} = float(T)
function TransformVariables.inverse_at!(x::AbstractVector, index,
        t::CompactVariogramMatrix, Γ::AbstractMatrix)
    size(Γ) == (t.d, t.d) || throw(DimensionMismatch("expected a $(t.d) × $(t.d) matrix"))
    n = TransformVariables.dimension(t)
    if _compact_is_zero_variogram(Γ)
        fill!(@view(x[index:(index + n - 1)]), -Inf)
        return index + n
    elseif _compact_is_independence_variogram(Γ)
        fill!(@view(x[index:(index + n - 1)]), Inf)
        return index + n
    end
    all(isfinite, Γ) || throw(DomainError(
        Γ, "variogram must contain only finite entries except for the independence boundary",
    ))
    return TransformVariables.inverse_at!(x, index, t.interior, Γ)
end

include("public_api.jl")
