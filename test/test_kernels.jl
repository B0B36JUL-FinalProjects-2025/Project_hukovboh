
using Test
using TVModels
using Distributions

@testset "Kernel Functions" begin
    @testset "epanechnikov_kernel" begin
        @test epanechnikov_kernel(0.0) ≈ 0.75
        @test epanechnikov_kernel(1.0) ≈ 0.0
        @test epanechnikov_kernel(2.0) ≈ 0.0
    end

    @testset "uniform_kernel" begin
        @test uniform_kernel(0.5) ≈ 0.5
        @test uniform_kernel(1.0) ≈ 0.5
        @test uniform_kernel(1.5) ≈ 0.0
    end

    @testset "triangular_kernel" begin
        @test triangular_kernel(0.0) ≈ 1.0
        @test triangular_kernel(0.5) ≈ 0.5
        @test triangular_kernel(1.5) ≈ 0.0
    end

    @testset "gaussian_kernel" begin
        @test gaussian_kernel(0.0) ≈ pdf(Normal(0, 1), 0.0)
        @test gaussian_kernel(1.0) ≈ pdf(Normal(0, 1), 1.0)
        @test gaussian_kernel(1.0) ≈ gaussian_kernel(-1.0)
    end

    @testset "quartic_kernel" begin
        @test quartic_kernel(0.0) ≈ 15/16
        @test quartic_kernel(1.0) ≈ 0.0
        @test quartic_kernel(1.5) ≈ 0.0
    end

    @testset "triweight_kernel" begin
        @test triweight_kernel(0.0) ≈ 35/32
        @test triweight_kernel(1.0) ≈ 0.0
        @test triweight_kernel(2.0) ≈ 0.0
    end

    @testset "tricube_kernel" begin
        @test tricube_kernel(0.0) ≈ 70/81
        @test tricube_kernel(1.0) ≈ 0.0
        @test tricube_kernel(1.5) ≈ 0.0
    end

    @testset "cosine_kernel" begin
        @test cosine_kernel(0.0) ≈ π/4
        @test cosine_kernel(1.5) ≈ 0.0
    end

    @testset "logistic_kernel" begin
        @test logistic_kernel(0.0) ≈ 1/4
        @test logistic_kernel(1.0) ≈ logistic_kernel(-1.0)
        @test logistic_kernel(5.0) > 0
    end

    @testset "sigmoid_kernel" begin
        @test sigmoid_kernel(0.0) ≈ 1/π
        @test sigmoid_kernel(1.0) ≈ sigmoid_kernel(-1.0)
        @test sigmoid_kernel(3.0) > 0
    end

    @testset "silverman_kernel" begin
        @test silverman_kernel(0.0) ≈ 1/2 * sin(π/4)
        @test silverman_kernel(1.0) ≈ silverman_kernel(-1.0)
        @test isfinite(silverman_kernel(2.0))
        @test silverman_kernel(5.0) < 0
    end

end