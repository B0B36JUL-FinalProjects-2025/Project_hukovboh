
using Test
using TVModels

@testset "Kernel Regressions" begin
    @testset "kernel_regression" begin
        # Generate synthetic data with known relationship
        T = 150
        Random.seed!(321)

        X = randn(T, 2)
        true_betas = [1.0, 0.5, -0.3]  # intercept + 2 features
        y = true_betas[1] .+ X * true_betas[2:end] + 0.1 * randn(T)
        
        bandwidth = 0.2
        
        # Basic functionality - multiple features
        betas_multi = kernel_regression(y, X, bandwidth)
        @test size(betas_multi) == (T, 3)  # intercept + 2 features
        @test isa(betas_multi, Matrix{Float64})
        
        # Check coefficients are reasonable (close to true values with small noise)
        @test all(abs.(reshape(mean(betas_multi, dims=1), 3) .- true_betas) .< 0.3)
        
        # Basic functionality - single feature
        X_single = X[:, 1]
        betas_single = kernel_regression(y, X_single, bandwidth)
        @test size(betas_single) == (T, 2)  # intercept + 1 feature
        
        # Test with different kernel
        betas_gaussian = kernel_regression(y, X, bandwidth; kernel=gaussian_kernel)
        @test size(betas_gaussian) == (T, 3)
        
        # Error cases
        @test_throws DimensionMismatch kernel_regression(y[1:100], X, bandwidth)  # Mismatched dimensions
        @test_throws DomainError kernel_regression(y, X, -0.1)  # Negative bandwidth
        @test_throws DomainError kernel_regression(randn(2), randn(2, 3), 0.5)  # Not enough data
        
        # Test perfect fit case (no noise)
        y_perfect = true_betas[1] .+ X * true_betas[2:end]
        betas_perfect = kernel_regression(y_perfect, X, bandwidth)
        @test all(abs.(reshape(mean(betas_perfect, dims=1), 3) .- true_betas) .< 1e-10)
    end

    @testset "kernel_ar" begin
        # Generate AR(2) process with known coefficients
        T = 200
        true_ar_coefs = [0.0, 0.6, -0.3]
        y = simulate_ar(T, true_ar_coefs; σ=0.05, seed=654)
        
        bandwidth = 0.15
        
        # Basic functionality
        betas_ar = kernel_ar(y, 2, bandwidth)
        @test size(betas_ar, 2) == 3  # intercept + 2 lags
        
        # Coefficients should vary over time (standard deviation > 0)
        @test std(betas_ar[:, 2]) > 0
        @test std(betas_ar[:, 3]) > 0
        
        # Error cases
        @test_throws DomainError kernel_ar(randn(2), 2, bandwidth)  # Not enough data
        @test_throws DomainError kernel_ar(y, 0, bandwidth)  # p must be positive
        @test_throws DomainError kernel_ar(y, 2, -0.2)  # Negative bandwidth

        # Test variability
        β_funcs = [t -> 0.5 + 0.2 * sin(2π * t), t -> -0.2 + 0.1 * cos(2π * t)]
        y_tv = simulate_tv_ar(T, β_funcs; σ=0.5, seed=222)
        betas_tv = kernel_ar(y_tv, 2, bandwidth)
        @test size(betas_tv, 1) == T - 2
        @test size(betas_tv, 2) == 3 
        @test std(betas_tv[:, 1]) > 0.05
        @test std(betas_tv[:, 2]) > 0.05
    end

    @testset "lowess_regression" begin    
        # Generate synthetic data with known relationship
        T = 150
        Random.seed!(321)

        X = randn(T, 2)
        true_betas = [1.0, 0.5, -0.3]  # intercept + 2 features
        y = true_betas[1] .+ X * true_betas[2:end] + 0.1 * randn(T)
        
        bandwidth = 0.2  
         
        # Basic functionality - multiple features
        betas_multi = lowess_regression(y, X, bandwidth, 3)
        @test size(betas_multi) == (T, 3)  # intercept + 2 features
        @test isa(betas_multi, Matrix{Float64})
        
        # Check coefficients are reasonable (close to true values with small noise)
        @test all(abs.(reshape(mean(betas_multi, dims=1), 3) .- true_betas) .< 0.3)
        
        # Basic functionality - single feature
        X_single = X[:, 1]
        betas_single = lowess_regression(y, X_single, bandwidth, 3)
        @test size(betas_single) == (T, 2)  # intercept + 1 feature
        
        # Test with different kernel
        betas_gaussian = lowess_regression(y, X, bandwidth, 3; kernel=gaussian_kernel)
        @test size(betas_gaussian) == (T, 3)
        
        # Error cases
        @test_throws DimensionMismatch lowess_regression(y[1:100], X, bandwidth, 3)  # Mismatched dimensions
        @test_throws DomainError lowess_regression(y, X, -0.1, 3)  # Negative bandwidth
        @test_throws DomainError lowess_regression(randn(2), randn(2, 3), 0.5, 3)  # Not enough data
        @test_throws DomainError lowess_regression(y, X, bandwidth, 0)  # Iterations must be positive
        
        # Test perfect fit case (no noise)
        y_perfect = true_betas[1] .+ X * true_betas[2:end]
        betas_perfect = lowess_regression(y_perfect, X, bandwidth, 3)
        @test all(abs.(reshape(mean(betas_perfect, dims=1), 3) .- true_betas) .< 1e-10)
    end

    @testset "lowess_ar" begin
        # Generate AR(2) process with known coefficients
        T = 200
        true_ar_coefs = [0.0, 0.6, -0.3]
        y = simulate_ar(T, true_ar_coefs; σ=0.05, seed=654)
        
        bandwidth = 0.15

        # Basic functionality
        betas_ar = lowess_ar(y, 2, bandwidth, 3)
        @test size(betas_ar, 2) == 3  # intercept + 2 lags
        
        # Coefficients should vary over time (standard deviation > 0)
        @test std(betas_ar[:, 2]) > 0
        @test std(betas_ar[:, 3]) > 0
        
        # Error cases
        @test_throws DomainError lowess_ar(randn(2), 2, bandwidth, 3)  # Not enough data
        @test_throws DomainError lowess_ar(y, 0, bandwidth, 3)  # p must be positive
        @test_throws DomainError lowess_ar(y, 2, -0.2, 3)  # Negative bandwidth
        @test_throws DomainError lowess_ar(y, 2, bandwidth, 0)  # Iterations must be positive

        # Test variability
        β_funcs = [t -> 0.5 + 0.2 * sin(2π * t), t -> -0.2 + 0.1 * cos(2π * t)]
        y_tv = simulate_tv_ar(T, β_funcs; σ=0.5, seed=222)
        betas_tv = lowess_ar(y_tv, 2, bandwidth, 3)
        @test size(betas_tv, 1) == T - 2
        @test size(betas_tv, 2) == 3 
        @test std(betas_tv[:, 1]) > 0.05
        @test std(betas_tv[:, 2]) > 0.05
    end
end