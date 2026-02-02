
using Test
using TVModels

@testset "Bayesian Regressions" begin
    @testset "QBLL_regression" begin
        # Simple linear regression test
        y = simulate_tv_ar(100, [t -> 1.0 + 0.6*cos(t), t -> 0.6*sin(t)]; σ=0.1, seed=123)
        Nsim = 1000
        H = 20

        betas = QBLL_ar(y, 1, Nsim, H; kernel=epanechnikov_kernel)
        @test size(betas) == (99, 2)

        # Varying betas over time
        y2 = simulate_tv_ar(150, [t -> 0.5 + 0.8*sin(2t), t -> -0.2 + 0.6*cos(3t)]; σ=0.05, seed=321)
        betas2 = QBLL_ar(y2, 1, 500, 25; kernel=epanechnikov_kernel)

        @test std(betas2[:, 1]) > 1e-3
        @test std(betas2[:, 2]) > 1e-3

        # Shape should be (T - p, p)
        @test size(betas2) == (150 - 1, 2)

        y_short = [1.0, 2.0]
        @test_throws DomainError QBLL_ar(y_short, 2, Nsim, H; kernel=epanechnikov_kernel)
        @test_throws DomainError QBLL_ar(y, -1, Nsim, H; kernel=epanechnikov_kernel)
        @test_throws DomainError QBLL_ar(y, 2, -1, H; kernel=epanechnikov_kernel)
        @test_throws DomainError QBLL_ar(y, 2, Nsim, -1; kernel=epanechnikov_kernel)
        @test_throws DomainError QBLL_ar(y, 2, -1, 1000; kernel=epanechnikov_kernel)
    end
end