
using Test
using TVModels

@testset "Forecast Models" begin
    @testset "forecast_regression" begin
        X = [1.0 2.0; 3.0 4.0]
        β = [0.5 0.1 0.2; 1.0 0.3 0.4]

        forecasts = forecast_regression(X, β)
        @test size(forecasts) == (2, 1)
        @test isapprox(forecasts[1], 0.5 + 0.1*1.0 + 0.2*2.0; atol=1e-8)
        @test isapprox(forecasts[2], 1.0 + 0.3*3.0 + 0.4*4.0; atol=1e-8)

        # Error Handling
        @test_throws DimensionMismatch forecast_regression([1.0, 2.0], [0.5, 0.1, 0.2, 0.3])
        @test_throws DimensionMismatch forecast_regression([1.0, 2.0], Float64[])
    end

    @testset "forecast_ar" begin
        X = [1.0, 2.0, 3.0]
        β = [0.5, 0.1, 0.2, 0.3]
        h = 2
        forecasts = forecast_ar(X, β, h)
        @test length(forecasts) == h
        @test isapprox(forecasts[1], 0.5 + 0.1*3.0 + 0.2*2.0 + 0.3*1.0; atol=1e-8)

        # Error Handling
        @test_throws DimensionMismatch forecast_ar([1.0], [0.5, 0.1, 0.3], 2)
        @test_throws DomainError forecast_ar([1.0, 2.0], Float64[], 2)
        @test_throws DomainError forecast_ar([1.0, 2.0], [0.5, 0.1], 0)
    end

    @testset "forecast_local_level" begin
        a = 5.0
        h = 3
        forecasts = forecast_local_level(a, h)
        @test length(forecasts) == h
        @test all(forecasts .== a)

        # Error Handling
        @test_throws DomainError forecast_local_level(a, 0)
    end
end