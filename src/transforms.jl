# Implementation type for `closed_lower`.
struct ClosedLower{L} <: TransformVariables.ScalarTransform
    lower::L
end

"""`closed_lower(lower)` creates a lower-bounded transform whose inverse accepts the boundary."""
closed_lower(lower) = ClosedLower(lower)
"""`nonnegative()` creates a transform to the closed nonnegative half-line."""
nonnegative() = ClosedLower(0)

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
