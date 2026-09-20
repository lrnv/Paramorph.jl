@testset "Copulas-derived geometry contracts" begin
    @testset "closed scalar domains" begin
        lower = closed_lower(1.0)
        @test TransformVariables.transform(lower, -Inf) == 1.0
        @test TransformVariables.inverse(lower, 1.0) == -Inf
        @test_throws DomainError TransformVariables.inverse(lower, 0.9)

        interval = bounded_interval(-1.0, 1.0)
        @test TransformVariables.inverse(interval, -1.0) == -Inf
        @test TransformVariables.inverse(interval, 1.0) == Inf
        @test TransformVariables.transform(interval, 0.0) == 0.0
        @test_throws DomainError TransformVariables.inverse(interval, 1.1)
    end

    @paramorph struct CorrelationLike{D}
        correlation::correlation_matrix(D)
    end

    @testset "Gaussian and Student correlation geometry" begin
        t = correlation_matrix(3)
        x = [0.2, -0.3, 0.4]
        R = TransformVariables.transform(t, x)
        @test R ≈ transpose(R)
        @test diag(R) ≈ ones(3)
        @test isposdef(R)
        @test TransformVariables.inverse(t, R) ≈ x
        R2, logjac = TransformVariables.transform_and_logjac(t, x)
        @test R2 ≈ R
        @test isfinite(logjac)
        prototype = constraint(CorrelationLike{3,Float64}, x)
        derivative = ForwardDiff.derivative(0.2) do z
            transformed = constraint(prototype, [z, -0.3, 0.4])
            transformed.correlation[1, 2]
        end
        @test isfinite(derivative)
    end

    @paramorph struct ContextChild
        value::closed_lower(get(context, :lower, -1.0))
    end

    @paramorph struct ContextParent{D}
        child::ContextChild{T}
    end

    Paramorph.schema_context(::Type{<:ContextParent{D}}, ::Val{:child}) where {D} =
        (; lower=-inv(D - 1))

    @testset "parent-dependent nested schema" begin
        @test ContextChild(-0.75).value == -0.75
        @test ContextParent{2}(ContextChild(-0.75)).child.value == -0.75
        @test_throws DomainError ContextParent{3}(ContextChild(-0.75))
        rebuilt = constraint(ContextParent{3,Float64}, [0.0])
        @test rebuilt.child.value ≈ 0.5
        prototype = ContextParent{3}(ContextChild(0.0))
        @test constraint(prototype, [0.0]).child.value ≈ 0.5
    end

    @paramorph T struct OrderedLike{T<:Real}
        lower::T
        upper::T
    end

    function Paramorph.schema_override(::Type{<:OrderedLike}, ::NamedTuple)
        base = TransformVariables.as((lower=asℝ, gap=asℝ₊))
        forward(x) = (; lower=x.lower, upper=x.lower + x.gap)
        backward(x) = (; lower=x.lower, gap=x.upper - x.lower)
        flatten(x) = [x.lower, x.upper]
        return joint_transform(base, forward, backward; flatten)
    end
    Paramorph.parameter_fields_override(::Type{<:OrderedLike}) = (:lower, :upper)

    @testset "joint field transformation" begin
        pair, logjac = constraint_with_logjac(OrderedLike{Float64}, [0.3, -0.2])
        @test pair.lower < pair.upper
        @test unconstrain(pair) ≈ [0.3, -0.2]
        @test isfinite(logjac)
        @test_throws DomainError OrderedLike(2.0, 1.0)
    end

    @paramorph T struct TawnLike{D,T<:Real}
        dep::as(Vector, closed_lower(one(T)), 2^D-D-1)
        weights::repeat_transform(UnitSimplex(2^(D-1)), D)
    end

    @paramorph T struct AsymGalambosLike{D,T<:Real}
        dep::as(Vector, nonnegative(), 2^D-D-1)
        weights::repeat_transform(UnitSimplex(2^(D-1)), D)
    end

    @testset "Tawn and asymmetric Galambos product geometry" begin
        for Family in (TawnLike, AsymGalambosLike)
            T = Family{3,Float64}
            expected = (2^3 - 3 - 1) + 3 * (2^(3 - 1) - 1)
            @test intrinsic_dimension(T) == expected
            object = constraint(T, zeros(expected))
            @test length(object.dep) == 4
            @test length(object.weights) == 3
            @test all(length(weight) == 4 for weight in object.weights)
            @test all(isapprox(sum(weight), 1.0) for weight in object.weights)
            @test unconstrain(object) ≈ zeros(expected) atol=10eps()
        end
        @test all(>=(1), constraint(TawnLike{3,Float64}, zeros(13)).dep)
        @test all(>=(0), constraint(AsymGalambosLike{3,Float64}, zeros(13)).dep)

        boundary = TawnLike{2}([1.0], [[0.0, 1.0], [0.25, 0.75]])
        @test boundary.dep == [1.0]
        @test boundary.weights[1] == [0.0, 1.0]
    end

    @paramorph struct ComponentLike
        strength::asℝ₊
    end

    @paramorph T struct LiebscherLike{D,T<:Real}
        components::recursive(Tuple{ComponentLike{T},ComponentLike{T}})
        weights::repeat_transform(UnitSimplex(2), D)
        label::String = "template"
    end

    @paramorph T struct VectorCompositeLike{N,T<:Real}
        children::recursive(Vector{ComponentLike{T}}, N)
    end

    @paramorph T struct NamedCompositeLike{T<:Real}
        children::recursive(NamedTuple{(:left,:right),Tuple{ComponentLike{T},ComponentLike{T}}})
    end

    @testset "recursive tuple, vector, and named tuple containers" begin
        prototype = LiebscherLike{2}(
            (ComponentLike(1.5), ComponentLike(2.0)),
            [[0.3, 0.7], [0.4, 0.6]],
            "kept",
        )
        x = unconstrain(prototype)
        @test length(x) == 4
        rebuilt = constraint(prototype, x .+ 0.1)
        @test rebuilt.label == "kept"
        @test rebuilt.components[1].strength ≈ 1.5exp(0.1)
        @test rebuilt.components[2].strength ≈ 2exp(0.1)
        @test all(isapprox(sum(w), 1) for w in rebuilt.weights)
        gradient = ForwardDiff.gradient(x) do coordinates
            candidate = constraint(prototype, coordinates)
            candidate.components[1].strength + candidate.components[2].strength
        end
        @test all(isfinite, gradient)

        vector_model = constraint(VectorCompositeLike{3,Float64}, zeros(3))
        @test length(vector_model.children) == 3
        @test all(child -> child.strength == 1, vector_model.children)

        named_model = constraint(NamedCompositeLike{Float64}, zeros(2))
        @test keys(named_model.children) == (:left, :right)
        @test named_model.children.left.strength == 1
        @test unconstrain(named_model) == zeros(2)
    end

    @paramorph T struct ActiveSimplexLike{T<:Real}
        weights::Vector{T}
        label::String = "active face"
    end
    Paramorph.parameter_fields_override(::Type{<:ActiveSimplexLike}) = (:weights,)
    function active_simplex_schema(weights)
        active = findall(!iszero, weights)
        isempty(active) && throw(DomainError(weights, "at least one weight must be active"))
        base = UnitSimplex(length(active))
        forward(values) = begin
            result = zeros(eltype(values), length(weights))
            result[active] = values
            result
        end
        backward(values) = values[active]
        flatten(values) = values[active[1:(end - 1)]]
        return TransformVariables.as((weights=joint_transform(
            base, forward, backward; flatten,
        ),))
    end
    Paramorph.schema_override(::Type{<:ActiveSimplexLike}, ::NamedTuple, values::NamedTuple) =
        active_simplex_schema(values.weights)
    Paramorph.schema_override(object::ActiveSimplexLike, ::NamedTuple) =
        active_simplex_schema(object.weights)

    @testset "prototype-dependent active simplex" begin
        prototype = ActiveSimplexLike([0.2, 0.0, 0.8], "kept")
        @test intrinsic_dimension(prototype) == 1
        x = unconstrain(prototype)
        rebuilt = constraint(prototype, x .+ 0.2)
        @test rebuilt.weights[2] == 0
        @test sum(rebuilt.weights) ≈ 1
        @test rebuilt.label == "kept"
    end

    function fgm_polytope(d)
        d == 3 || throw(ArgumentError("this reduced fixture is defined for d = 3"))
        subsets = [(1, 2), (1, 3), (2, 3), (1, 2, 3)]
        corners = collect(Iterators.product(ntuple(_ -> (-1.0, 1.0), d)...))
        coefficients = reduce(vcat, [
            reshape([-prod(corner[i] for i in subset) for subset in subsets], 1, :)
            for corner in corners
        ])
        q = length(subsets)
        A = [coefficients; Matrix{Float64}(I, q, q); -Matrix{Float64}(I, q, q)]
        return polytope(A, ones(size(A, 1)))
    end

    @paramorph T struct FGMPolytopeLike{T<:Real}
        theta::Vector{T}
    end
    Paramorph.parameter_fields_override(::Type{<:FGMPolytopeLike}) = (:theta,)
    Paramorph.schema_override(::Type{<:FGMPolytopeLike}, ::NamedTuple) =
        TransformVariables.as((theta=fgm_polytope(3),))

    @testset "multivariate FGM polytope escape hatch" begin
        fgm = fgm_polytope(3)
        @test TransformVariables.dimension(fgm) == 4
        x = [0.2, -0.4, 0.1, 0.3]
        model, logjac = constraint_with_logjac(FGMPolytopeLike{Float64}, x)
        @test unconstrain(model) ≈ x
        @test all(fgm.A * model.theta .<= fgm.b)
        @test isfinite(logjac)
        @test_throws DomainError FGMPolytopeLike(fill(2.0, 4))
    end
end
