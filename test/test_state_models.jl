
using Test
using TVModels
using LinearAlgebra
using Random

@testset "State Space Models" begin

    @testset "State Local Level Model" begin
        Random.seed!(42)
        T = 50
        y = randn(T)
        
        a_filt, P_filt, a_smooth, P_smooth = state_local_level(y; 
            sigma_eta=0.1, sigma_eps=0.5, a0=0.0, P0=1.0, smooth=true)
        
        # Test proper output sizes and types
        @test length(a_filt) == T
        @test length(P_filt) == T
        @test length(a_smooth) == T
        @test length(P_smooth) == T

        @test isa(a_filt, Vector{Float64})
        @test isa(P_filt, Vector{Float64})
        @test isa(a_smooth, Vector{Float64})
        @test isa(P_smooth, Vector{Float64})

        # Smoothed estimates have lower variance than filtered
        @test mean(P_smooth) < mean(P_filt)

        # High state noise: filter trusts observations
        a_high_eta, _ = state_local_level(y; 
        sigma_eta=1.0, sigma_eps=0.01, a0=0.0, P0=1.0, smooth=false)
        
        # Low state noise: filter trusts prior
        a_low_eta, _ = state_local_level(y; 
        sigma_eta=0.01, sigma_eps=1.0, a0=0.0, P0=1.0, smooth=false)
        
        # High state noise estimates should be closer to observations
        @test mean(abs.(a_high_eta .- y)) < mean(abs.(a_low_eta .- y))
        
        # Error handling
        @test_throws DomainError state_local_level([1.0], sigma_eta=-0.1, sigma_eps=0.5, a0=0.0, P0=1.0)  # Too few observations
        @test_throws DomainError state_local_level(y; sigma_eta=-0.1, sigma_eps=0.5, a0=0.0, P0=1.0) # Negative sigma_eta
        @test_throws DomainError state_local_level(y; sigma_eta=0.0, sigma_eps=0.5, a0=0.0, P0=1.0) # Zero sigma_eta
        @test_throws DomainError state_local_level(y; sigma_eta=0.1, sigma_eps=-0.5, a0=0.0, P0=1.0) # Negative sigma_eps
        @test_throws DomainError state_local_level(y; sigma_eta=0.1, sigma_eps=0.0, a0=0.0, P0=1.0) # Zero sigma_eps
        @test_throws DomainError state_local_level(y; sigma_eta=0.1, sigma_eps=0.5, a0=0.0, P0=-1.0) # Negative P0
        @test_throws DomainError state_local_level(y; sigma_eta=0.1, sigma_eps=0.5, a0=0.0, P0=0.0) # Zero P0
    end

    @testset "State AR Model" begin
        # Generate AR(1) data
        T = 100
        p = 1

        y_ar = simulate_ar(T, [0.0, 0.7]; σ=0.5, seed=42)
        
        β_filt, P_filt, β_smooth, P_smooth = state_ar(y_ar, p;
            sigma_eps=0.5, Q=0.01, β0=[0.0, 0.5], P0=1.0, smooth=true)
        
        @test size(β_filt) == (T-p, p+1)
        @test size(P_filt) == (T-p, p+1, p+1)
        @test size(β_smooth) == (T-p, p+1)
        @test size(P_smooth) == (T-p, p+1, p+1)

        # Test different input formats for Q
        β1, _ = state_ar(y_ar, p; sigma_eps=0.5, Q=0.01, β0=[0.0, 0.5], P0=1.0, smooth=false)
        β2, _ = state_ar(y_ar, p; sigma_eps=0.5, Q=[0.01, 0.01], β0=[0.0, 0.5], P0=1.0, smooth=false)
        β3, _ = state_ar(y_ar, p; sigma_eps=0.5, Q=0.01*I(p+1), β0=[0.0, 0.5], P0=1.0, smooth=false)
        
        @test β1 ≈ β2 rtol=1e-10
        @test β1 ≈ β3 rtol=1e-10

        # Test different input formats for P0
        β1, _ = state_ar(y_ar, p; sigma_eps=0.5, Q=0.01, β0=[0.0, 0.5], P0=1.0, smooth=false)
        β2, _ = state_ar(y_ar, p; sigma_eps=0.5, Q=0.01, β0=[0.0, 0.5], P0=[1.0, 1.0], smooth=false)
        β3, _ = state_ar(y_ar, p; sigma_eps=0.5, Q=0.01, β0=[0.0, 0.5], P0=I(p+1), smooth=false)
        
        @test β1 ≈ β2 rtol=1e-10
        @test β1 ≈ β3 rtol=1e-10

        # Higher order AR
        p_high = 3
        β_filt, P_filt = state_ar(y_ar, p_high;
            sigma_eps=0.5, Q=0.01, β0=zeros(p_high+1), P0=1.0, smooth=false)
        
        @test size(β_filt) == (T-p_high, p_high+1)

        # Custom state transition (AR(2))
        ρ₁, ρ₂ = 0.6, 0.2
        ar2_trans = (β_hist, P_hist, Q) -> begin
            if size(β_hist, 1) < 2
                return (β_hist[end, :], P_hist[end, :, :] + Q)
            else
                β_new = ρ₁ * β_hist[end, :] + ρ₂ * β_hist[end-1, :]
                P_new = dropdims(sum(P_hist[end-1:end, :, :], dims=1), dims=1)
                return (β_new, P_new + Q)
            end
        end
        
        β_filt, P_filt = state_ar(y_ar, p;
            sigma_eps=0.5, Q=0.01, β0=[0.0, 0.5], P0=1.0, 
            state_transition=ar2_trans, smooth=false)
        
        @test size(β_filt) == (T-p, p+1)

        # Smoothed estimates have lower variance than filtered
        β_filt, P_filt, β_smooth, P_smooth = state_ar(y_ar, p;
        sigma_eps=0.5, Q=0.01, β0=[0.0, 0.5], P0=1.0, smooth=true)
        
        @test mean(P_smooth) < mean(P_filt)

        # Coefficients converge over time
        β_filt, P_filt = state_ar(y_ar, p;
        sigma_eps=0.5, Q=0.001, β0=[0.0, 0.5], P0=1.0, smooth=false)
        
        # With low Q, covariance should decrease
        @test mean(P_filt[50:end, :, :]) < mean(P_filt[1:10, :, :])

        # Error handling
        @test_throws DomainError state_ar([1.0, 2.0], 2; sigma_eps=0.5, Q=0.01, β0=[0.0, 0.0, 0.0], P0=1.0) # Too few observations
        @test_throws DomainError state_ar(y_ar, 0; sigma_eps=0.5, Q=0.01, β0=[0.0, 0.5], P0=1.0) # p < 1
        @test_throws DomainError state_ar(y_ar, p; sigma_eps=0.5, Q=0.01, β0=[0.0], P0=1.0) # Incorrect β0 size
        @test_throws DomainError state_ar(y_ar, p; sigma_eps=0.5, Q=zeros(3,3), β0=[0.0, 0.5], P0=1.0) # Incorrect Q size
        @test_throws DomainError state_ar(y_ar, p; sigma_eps=-0.5, Q=0.01, β0=[0.0, 0.5], P0=1.0) # Negative sigma_eps
        @test_throws DomainError state_ar(y_ar, p; sigma_eps=0.5, Q=-0.01, β0=[0.0, 0.5], P0=1.0) # Negative Q
        @test_throws DomainError state_ar(y_ar, p; sigma_eps=0.5, Q=0.01, β0=[0.0, 0.5], P0=-1.0) # Negative P0
    end

    @testset "State Regression Model" begin
        Random.seed!(42)
        T = 100
        X = randn(T, 2)
        y = 1.0 .+ 0.5 * X[:, 1] .- 0.3 * X[:, 2] + 0.2 * randn(T)
        
        β_filt, P_filt, β_smooth, P_smooth = state_regression(y, X; 
            sigma_eps=0.3, Q=0.01, β0=zeros(3), P0=1.0, smooth=true)
        
        # Test proper output sizes and types
        @test size(β_filt) == (T, 3)
        @test size(P_filt) == (T, 3, 3)
        @test size(β_smooth) == (T, 3)
        @test size(P_smooth) == (T, 3, 3)

        @test isa(β_filt, Matrix{Float64})
        @test isa(P_filt, Array{Float64, 3})
        @test isa(β_smooth, Matrix{Float64})
        @test isa(P_smooth, Array{Float64, 3})

        # Smoothed estimates have lower variance than filtered
        @test mean(P_smooth) < mean(P_filt)

        # Test different input formats for Q
        β1, _ = state_regression(y, X; sigma_eps=0.3, Q=0.01, β0=zeros(3), P0=1.0, smooth=false)
        β2, _ = state_regression(y, X; sigma_eps=0.3, Q=0.01*ones(3), β0=zeros(3), P0=1.0, smooth=false)
        β3, _ = state_regression(y, X; sigma_eps=0.3, Q=0.01*I(3), β0=zeros(3), P0=1.0, smooth=false)
        
        @test β1 ≈ β2 rtol=1e-10
        @test β1 ≈ β3 rtol=1e-10

        # Test different input formats for P0
        β1, _ = state_regression(y, X; sigma_eps=0.3, Q=0.01, β0=zeros(3), P0=1.0, smooth=false)
        β2, _ = state_regression(y, X; sigma_eps=0.3, Q=0.01, β0=zeros(3), P0=ones(3), smooth=false)
        β3, _ = state_regression(y, X; sigma_eps=0.3, Q=0.01, β0=zeros(3), P0=I(3), smooth=false)
        
        @test β1 ≈ β2 rtol=1e-10
        @test β1 ≈ β3 rtol=1e-10

        # High state noise: filter trusts observations
        β_high_Q, _ = state_regression(y, X; 
        sigma_eps=0.01, Q=0.1, β0=zeros(3), P0=1.0, smooth=false)
        
        # Low state noise: filter trusts prior
        β_low_Q, _ = state_regression(y, X; 
        sigma_eps=1.0, Q=0.001, β0=zeros(3), P0=1.0, smooth=false)
        
        # High state noise estimates should be closer to observations (in the sense of larger variation)
        @test mean(abs.(β_high_Q .- β_low_Q)) > 0
        
        # Error handling
        @test_throws DomainError state_regression(y[1:2], X[1:2, :]; sigma_eps=0.3, Q=0.01, β0=zeros(3), P0=1.0) # Too few observations
        @test_throws DomainError state_regression(y, X; sigma_eps=-0.3, Q=0.01, β0=zeros(3), P0=1.0) # Negative sigma_eps
        @test_throws DomainError state_regression(y, X; sigma_eps=0.0, Q=0.01, β0=zeros(3), P0=1.0) # Zero sigma_eps
        @test_throws DomainError state_regression(y, X; sigma_eps=0.3, Q=-0.01, β0=zeros(3), P0=1.0) # Negative Q
        @test_throws DomainError state_regression(y, X; sigma_eps=0.3, Q=0.0, β0=zeros(3), P0=1.0) # Zero Q
        @test_throws DomainError state_regression(y, X; sigma_eps=0.3, Q=0.01, β0=zeros(3), P0=-1.0) # Negative P0
        @test_throws DomainError state_regression(y, X; sigma_eps=0.3, Q=0.01, β0=zeros(3), P0=0.0) # Zero P0
    end
end