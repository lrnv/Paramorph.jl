# Sequential parameter products with scalar bounds depending on parameters
# constrained earlier in the same product. These spaces are deliberately
# small: they model ordered/dependent scalar domains without introducing a
# general constraint solver.

abstract type AbstractDependentScalarSpace <: AbstractParameterSpace end

struct GreaterThan{S,R,L,U} <: AbstractDependentScalarSpace
    lower::L
    upper::U
end

struct LowerThan{S,R,L,U} <: AbstractDependentScalarSpace
    lower::L
    upper::U
end

"""
    DependentProduct(spaces...)

A sequential product of parameter spaces in which later scalar parameters may
reference values constrained earlier in the same product through [`GreaterThan`](@ref)
or [`LowerThan`](@ref).

Unlike an ordinary tuple product, a `DependentProduct` keeps a small constrained-value
context while walking its component spaces. References must point to an earlier
logical parameter name, and logical names must be unique within the product.
"""
struct DependentProduct{T<:Tuple} <: AbstractParameterSpace
    spaces::T

    function DependentProduct{T}(spaces::T) where {T<:Tuple}
        all(q -> q isa AbstractParameterSpace, spaces) ||
            throw(ArgumentError("all dependent-product entries must be parameter spaces"))
        seen = Set{Symbol}()
        for q in spaces
            if q isa AbstractDependentScalarSpace
                ref = _dependent_reference(q)
                ref in seen || throw(ArgumentError(
                    "dependent parameter $(only(names(q))) references $ref before it is defined",
                ))
            end
            for name in names(q)
                name in seen && throw(ArgumentError(
                    "dependent products require unique logical parameter names; duplicate $name",
                ))
                push!(seen, name)
            end
        end
        return new{T}(spaces)
    end
end

"""
    GreaterThan(name, reference; lower=nothing, upper=nothing)

A scalar parameter `name` constrained to be greater than or equal to the
previously constrained scalar parameter `reference`. Optional intrinsic
bounds are intersected with that relation, so the effective interval is
`[max(reference, lower), upper]` when both are supplied.

`GreaterThan` is evaluated inside a [`DependentProduct`](@ref).
"""
GreaterThan(name::Symbol, reference::Symbol; lower=nothing, upper=nothing) =
    GreaterThan{name,reference,typeof(lower),typeof(upper)}(lower, upper)

"""
    LowerThan(name, reference; lower=nothing, upper=nothing)

A scalar parameter `name` constrained to be less than or equal to the
previously constrained scalar parameter `reference`. Optional intrinsic
bounds are intersected with that relation, so the effective interval is
`[lower, min(reference, upper)]` when both are supplied.

`LowerThan` is evaluated inside a [`DependentProduct`](@ref).
"""
LowerThan(name::Symbol, reference::Symbol; lower=nothing, upper=nothing) =
    LowerThan{name,reference,typeof(lower),typeof(upper)}(lower, upper)

_dependent_reference(::GreaterThan{S,R}) where {S,R} = R
_dependent_reference(::LowerThan{S,R}) where {S,R} = R

names(::GreaterThan{S}) where {S} = (S,)
names(::LowerThan{S}) where {S} = (S,)
dimension(::AbstractDependentScalarSpace) = 1

DependentProduct(spaces::Tuple) = DependentProduct{typeof(spaces)}(spaces)
DependentProduct(spaces::AbstractParameterSpace...) = DependentProduct(spaces)

names(p::DependentProduct) = Tuple(name for q in p.spaces for name in names(q))
dimension(p::DependentProduct) = sum(dimension, p.spaces; init=0)

function constrain(p::AbstractDependentScalarSpace, θ)
    throw(ArgumentError(
        "$(nameof(typeof(p))) depends on another constrained parameter and must be used inside DependentProduct",
    ))
end

function unconstrain(p::AbstractDependentScalarSpace, η)
    throw(ArgumentError(
        "$(nameof(typeof(p))) depends on another constrained parameter and must be used inside DependentProduct",
    ))
end

function _dependent_lookup(context, ref::Symbol)
    for (name, value) in context
        name === ref && return value
    end
    throw(ArgumentError("dependent parameter reference $ref is unavailable"))
end

function _effective_bounds(p::GreaterThan, context)
    ref = _dependent_lookup(context, _dependent_reference(p))
    ref isa Number || throw(ArgumentError("dependent scalar reference must be numeric"))
    lo = p.lower === nothing ? ref : max(ref, p.lower)
    return lo, p.upper
end

function _effective_bounds(p::LowerThan, context)
    ref = _dependent_lookup(context, _dependent_reference(p))
    ref isa Number || throw(ArgumentError("dependent scalar reference must be numeric"))
    hi = p.upper === nothing ? ref : min(ref, p.upper)
    return p.lower, hi
end

function _constrain_interval(lo, hi, z)
    if lo !== nothing && hi !== nothing
        lo <= hi || throw(DomainError((lo, hi), "dependent bounds define an empty interval"))
        lo == hi && return lo + zero(z)
        q = _constrain_scalar(ProbabilityDomain{true,true}(), z)
        return lo + (hi - lo) * q
    elseif lo !== nothing
        return lo + exp(z)
    elseif hi !== nothing
        return hi - exp(z)
    end
    return z
end

function _unconstrain_interval(lo, hi, η)
    if lo !== nothing && hi !== nothing
        lo <= hi || throw(DomainError((lo, hi), "dependent bounds define an empty interval"))
        lo <= η <= hi || throw(DomainError(η, "parameter lies outside its dependent interval"))
        lo == hi && return zero(float(η))
        q = (η - lo) / (hi - lo)
        return _unconstrain_scalar(ProbabilityDomain{true,true}(), q)
    elseif lo !== nothing
        η >= lo || throw(DomainError(η, "parameter must be greater than or equal to $lo"))
        return log(η - lo)
    elseif hi !== nothing
        η <= hi || throw(DomainError(η, "parameter must be less than or equal to $hi"))
        return log(hi - η)
    end
    return η
end

function _constrain_with_context(p::AbstractParameterSpace, θ, context)
    return constrain(p, θ)
end

function _constrain_with_context(p::AbstractDependentScalarSpace, θ, context)
    _check_dimension(p, θ)
    lo, hi = _effective_bounds(p, context)
    return _constrain_interval(lo, hi, θ[1])
end

function _unconstrain_with_context(p::AbstractParameterSpace, η, context)
    return unconstrain(p, η)
end

function _unconstrain_with_context(p::AbstractDependentScalarSpace, η, context)
    η isa Number || throw(ArgumentError("dependent scalar parameter must be numeric"))
    lo, hi = _effective_bounds(p, context)
    return [_unconstrain_interval(lo, hi, η)]
end

function _extend_dependent_context(context, q, ηq)
    qnames = names(q)
    qvalues = _parameter_values(q, ηq)
    return (context..., ntuple(i -> qnames[i] => qvalues[i], length(qnames))...)
end

function constrain(p::DependentProduct, θ)
    _check_dimension(p, θ)
    context = ()
    values = Any[]
    offset = 0
    for q in p.spaces
        n = dimension(q)
        ηq = _constrain_with_context(q, view(θ, (offset + 1):(offset + n)), context)
        append!(values, _parameter_values(q, ηq))
        context = _extend_dependent_context(context, q, ηq)
        offset += n
    end
    return Tuple(values)
end

function unconstrain(p::DependentProduct, η::Tuple)
    length(η) == length(names(p)) || throw(DimensionMismatch(
        "expected $(length(names(p))) constrained parameter values, got $(length(η))",
    ))
    context = ()
    pieces = Any[]
    offset = 0
    for q in p.spaces
        n = _parameter_count(q)
        qvalues = ntuple(i -> η[offset + i], n)
        ηq = _from_parameter_values(q, qvalues)
        push!(pieces, _unconstrain_with_context(q, ηq, context))
        context = _extend_dependent_context(context, q, ηq)
        offset += n
    end
    return _concatenate_vectors(pieces)
end


# ---------------------------------------------------------------------------
# Hot-path specializations
# ---------------------------------------------------------------------------
# These more-specific methods preserve the public transformations above while
# avoiding avoidable temporary objects in optimizer/AD loops.  The generic
# implementations in Paramorph.jl remain the fallback for unusual containers.

@inline function _constrain_product_piece(
    q::ScalarSpace,
    θ::AbstractVector,
    offset::Int,
)
    @inbounds z = θ[offset + 1]
    return _constrain_scalar(q.domain, z)
end

@inline function _constrain_product_piece(
    q::AbstractParameterSpace,
    θ::AbstractVector,
    offset::Int,
)
    n = dimension(q)
    return constrain(q, @view θ[(offset + 1):(offset + n)])
end

@inline _constrain_product_vector(::Tuple{}, ::AbstractVector, ::Int) = ()

@inline function _constrain_product_vector(
    p::Tuple,
    θ::AbstractVector,
    offset::Int,
)
    q = first(p)
    ηq = _constrain_product_piece(q, θ, offset)
    return (
        _parameter_values(q, ηq)...,
        _constrain_product_vector(Base.tail(p), θ, offset + dimension(q))...,
    )
end

function constrain(p::ProductParameterSpace, θ::AbstractVector)
    _check_dimension(p, θ)
    return _constrain_product_vector(p, θ, 0)
end

@inline function correlation_factor(
    p::Correlation,
    θ::AbstractVector{T},
) where {T<:Number}
    _check_dimension(p, θ)
    L = zeros(T, p.n, p.n)
    q = 0

    @inbounds for i in 1:p.n
        scale = one(T)
        for j in 1:(i - 1)
            q += 1
            z = tanh(θ[q])
            L[i, j] = scale * z
            scale *= sqrt(one(z) - z * z)
        end
        L[i, i] = scale
    end

    return L
end
