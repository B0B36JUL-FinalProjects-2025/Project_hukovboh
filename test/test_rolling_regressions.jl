
using Test
using TVModels
using LinearAlgebra
using Random

@testset "Rolling Regressions" begin
    @testset "rolling_regression" begin
        # Generate synthetic data with known relationship
        T = 100
        Random.seed!(123)
        X = randn(T, 2)
        true_betas = [1.0, 0.5, -0.3]  # intercept + 2 features
        y = true_betas[1] .+ X * true_betas[2:end] + 0.1 * randn(T)
        
        window_size = 30
        
        # Basic functionality - fixed window
        betas_fixed = rolling_regression(y, X, window_size; expanding=false)
        @test size(betas_fixed) == (T - window_size + 1, 3)
        @test isa(betas_fixed, Matrix{Float64})
        
        # Check coefficients are reasonable (close to true values with small noise)
        @test all(abs.(reshape(mean(betas_fixed, dims=1), 3) .- true_betas) .< 0.5)
        
        # Basic functionality - expanding window
        betas_expanding = rolling_regression(y, X, window_size; expanding=true)
        @test size(betas_expanding) == (T - window_size + 1, 3)
        
        # Expanding window should have same first column as fixed window
        @test betas_expanding[1, :] ≈ betas_fixed[1, :]
        
        # Test with single feature (multiple dispatch)
        X_single = X[:, 1]
        betas_single = rolling_regression(y, X_single, window_size)
        @test size(betas_single) == (T - window_size + 1, 2)  # intercept + 1 feature
        
        # Test with smaller window
        small_window = 20
        betas_small = rolling_regression(y, X, small_window)
        @test size(betas_small) == (T - small_window + 1, 3)
        @test size(betas_small, 1) > size(betas_fixed, 1)  # More windows with smaller window size
        
        # Error cases
        @test_throws DimensionMismatch rolling_regression(y[1:50], X, window_size)  # Mismatched dimensions
        @test_throws DomainError rolling_regression(y, X, T + 10)  # Window too large
        @test_throws DomainError rolling_regression(y, X, 2)  # Window too small for features
        @test_throws DomainError rolling_regression(randn(5), randn(5, 3), 10)  # Not enough data
        
        # Test perfect fit case (no noise)
        y_perfect = true_betas[1] .+ X * true_betas[2:end]
        betas_perfect = rolling_regression(y_perfect, X, window_size)
        @test all(abs.(reshape(mean(betas_perfect, dims=1), 3) .- true_betas) .< 1e-10)
    end

    @testset "rolling_ar" begin
        # Generate AR(2) process with known coefficients
        T = 200
        true_ar_coefs = [0.0, 0.6, -0.3]
        y = simulate_ar(T, true_ar_coefs; σ=0.05, seed=456)
        
        p = 2
        window_size = 50
        
        # Basic functionality - fixed window
        betas_fixed = rolling_ar(y, p, window_size; expanding=false)
        @test size(betas_fixed, 2) == p + 1  # intercept + p lags
        @test isa(betas_fixed, Matrix{Float64})
        
        # Check that estimated coefficients are close to true values (on average)
        mean_coefs = mean(betas_fixed, dims=1)
        @test all(abs.(reshape(mean_coefs, 3) .- true_ar_coefs) .< 0.3)  # Reasonable tolerance
        
        # Basic functionality - expanding window
        betas_expanding = rolling_ar(y, p, window_size; expanding=true)
        @test size(betas_expanding, 2) == p + 1
        
        # First window should be the same
        @test betas_expanding[1, :] ≈ betas_fixed[1, :]
        
        # Test with different window sizes
        small_window = 30
        betas_small = rolling_ar(y, p, small_window)
        @test size(betas_small, 2) == p + 1
        @test size(betas_small, 1) > size(betas_fixed, 1)  # More windows
        
        # Error cases
        @test_throws DomainError rolling_ar(y, 0, window_size)  # p < 1
        @test_throws DomainError rolling_ar(y, p, T + 10)  # Window too large
        @test_throws DomainError rolling_ar(y, p, 2)  # Window too small
        @test_throws DomainError rolling_ar(randn(5), 3, 10)  # Not enough data
        @test_throws DomainError rolling_ar(y, p, p)  # Window size exactly p (need p+1)
        
        # Test with time-varying AR process
        β_funcs = [t -> 0.5 + 0.2 * sin(2π * t), t -> -0.2]
        y_tv = simulate_tv_arp(T, β_funcs; σ=0.5, seed=222)
        betas_tv = rolling_ar(y_tv, 2, window_size)
        @test size(betas_tv, 2) == 3  # intercept + 2 lags
        # Coefficients should vary over time (standard deviation > 0)
        @test std(betas_tv[:, 2]) > 0.05
        
        # Test expanding window produces different results than fixed
        @test !(betas_expanding[end, :] ≈ betas_fixed[end, :])
    end
end
