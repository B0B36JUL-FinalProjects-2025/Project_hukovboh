
using Test
using TVModels

@testset "Simulate Series" begin
    @testset "simulate_tv_arp_matrix" begin
        # Basic functionality
        T = 100
        p = 2
        βs = repeat([0.5, -0.2]', T)
        
        y = simulate_tv_arp(T, βs; σ=1.0, seed=123)
        @test length(y) == T
        @test y isa Vector{Float64}
        
        # Test reproducibility with seed
        y1 = simulate_tv_arp(T, βs; σ=1.0, seed=42)
        y2 = simulate_tv_arp(T, βs; σ=1.0, seed=42)
        @test y1 == y2
        
        # Test different seed produces different results
        y3 = simulate_tv_arp(T, βs; σ=1.0, seed=43)
        @test y1 != y3
        
        # Test zero noise
        y_zero = simulate_tv_arp(T, βs; σ=0.0, seed=123)
        @test all(abs.(y_zero) .< 1e-10)  
        
        # Error cases
        @test_throws DomainError simulate_tv_arp(2, βs; σ=1.0)  # T < p + 1
        @test_throws DomainError simulate_tv_arp(T, βs; σ=-1.0)  # Negative σ
        @test_throws DimensionMismatch simulate_tv_arp(T, βs[1:50, :]; σ=1.0)  # Wrong size
        @test_throws DomainError simulate_tv_arp(T, zeros(0, 0); σ=1.0)  # Empty matrix
    end

    @testset "simulate_tv_arp_functions" begin
        # Basic functionality with constant functions
        T = 100
        β_funcs = [t -> 0.5, t -> -0.2]
        
        y = simulate_tv_arp(T, β_funcs; σ=1.0, seed=123)
        @test length(y) == T
        @test y isa Vector{Float64}
        
        # Test with time-varying functions
        β_funcs_tv = [t -> 0.5 * sin(2π * t), t -> -0.2 * cos(2π * t)]
        y_tv = simulate_tv_arp(T, β_funcs_tv; σ=1.0, seed=456)
        @test length(y_tv) == T
        
        # Test reproducibility
        y1 = simulate_tv_arp(T, β_funcs; σ=1.0, seed=789)
        y2 = simulate_tv_arp(T, β_funcs; σ=1.0, seed=789)
        @test y1 == y2
        
        # Test that function version matches matrix version for constant functions
        βs_matrix = repeat([0.5, -0.2]', T)
        y_matrix = simulate_tv_arp(T, βs_matrix; σ=1.0, seed=100)
        y_func = simulate_tv_arp(T, β_funcs; σ=1.0, seed=100)
        @test y_matrix ≈ y_func
        
        # Error handling - function that throws
        bad_funcs = [t -> 0.5, t -> error("Test error")]
        @test_throws ArgumentError simulate_tv_arp(T, bad_funcs; σ=1.0)
        
        # Function that returns wrong type
        type_error_funcs = [t -> "string", t -> 0.5]
        @test_throws ArgumentError simulate_tv_arp(T, type_error_funcs; σ=1.0)
    end

    @testset "simulate_ar" begin
        # Basic functionality
        T = 100
        βs = [0.5, -0.2]
        
        y = simulate_ar(T, βs; σ=1.0, seed=123)
        @test length(y) == T
        @test y isa Vector{Float64}
        
        # Test reproducibility with seed
        y1 = simulate_ar(T, βs; σ=1.0, seed=42)
        y2 = simulate_ar(T, βs; σ=1.0, seed=42)
        @test y1 == y2
        
        # Test different seed produces different results
        y3 = simulate_ar(T, βs; σ=1.0, seed=43)
        @test y1 != y3
        
        # Test zero noise
        y_zero = simulate_ar(T, βs; σ=0.0, seed=123)
        @test all(abs.(y_zero) .< 1e-10)  # Should be near zero with zero noise
        
        # Test AR(1) process
        β_ar1 = [0.8]
        y_ar1 = simulate_ar(200, β_ar1; σ=1.0, seed=999)
        @test length(y_ar1) == 200
        
        # Error cases
        @test_throws DomainError simulate_ar(2, βs; σ=1.0)  # T < p + 1
        @test_throws DomainError simulate_ar(T, βs; σ=-1.0)  # Negative σ
        @test_throws DomainError simulate_ar(T, Float64[]; σ=1.0)  # Empty vector
        
        # Test with different coefficient values
        βs_stable = [0.3, 0.3]
        y_stable = simulate_ar(T, βs_stable; σ=1.0, seed=555)
        @test length(y_stable) == T
    end
end