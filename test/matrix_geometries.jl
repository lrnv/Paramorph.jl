@testset "closed correlation matrix geometry" begin
    geometry = closed_correlation_matrix(3)
    @test TV.dimension(geometry) == 3

    interior = TV.transform(geometry, zeros(3))
    @test size(interior) == (3, 3)
    @test all(isfinite, interior)
    @test TV.transform(geometry, TV.inverse(geometry, interior)) ≈ interior

    complete = ones(3, 3)
    complete_coordinates = TV.inverse(geometry, complete)
    @test all(isinf, complete_coordinates)
    @test all(>(0), complete_coordinates)
    @test TV.transform(geometry, complete_coordinates) == complete
    complete_value, complete_logjac =
        TV.transform_and_logjac(geometry, complete_coordinates)
    @test complete_value == complete
    @test complete_logjac == -Inf

    @test_throws DimensionMismatch TV.inverse(geometry, ones(2, 2))
    @test_throws DomainError TV.inverse(geometry, [1.0 1.0 0.0; 1.0 1.0 0.0; 0.0 0.0 1.0])
    @test_throws DomainError TV.inverse(geometry, [1.0 Inf 0.0; Inf 1.0 0.0; 0.0 0.0 1.0])
end

@testset "compact variogram matrix geometry" begin
    geometry = compact_variogram_matrix(3)
    @test TV.dimension(geometry) == 3

    interior = TV.transform(geometry, zeros(3))
    @test size(interior) == (3, 3)
    @test all(isfinite, interior)
    @test TV.transform(geometry, TV.inverse(geometry, interior)) ≈ interior

    complete = zeros(3, 3)
    complete_coordinates = TV.inverse(geometry, complete)
    @test all(isinf, complete_coordinates)
    @test all(<(0), complete_coordinates)
    @test TV.transform(geometry, complete_coordinates) == complete
    complete_value, complete_logjac =
        TV.transform_and_logjac(geometry, complete_coordinates)
    @test complete_value == complete
    @test complete_logjac == -Inf

    independence = fill(Inf, 3, 3)
    for i in axes(independence, 1)
        independence[i, i] = 0.0
    end
    independence_coordinates = TV.inverse(geometry, independence)
    @test all(isinf, independence_coordinates)
    @test all(>(0), independence_coordinates)
    @test TV.transform(geometry, independence_coordinates) == independence
    independence_value, independence_logjac =
        TV.transform_and_logjac(geometry, independence_coordinates)
    @test independence_value == independence
    @test independence_logjac == -Inf

    @test_throws DimensionMismatch TV.inverse(geometry, zeros(2, 2))
    mixed_infinite = copy(independence)
    mixed_infinite[1, 2] = mixed_infinite[2, 1] = 1.0
    @test_throws DomainError TV.inverse(geometry, mixed_infinite)
end
