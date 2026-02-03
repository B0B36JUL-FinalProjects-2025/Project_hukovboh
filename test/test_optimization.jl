
using Test
using TVModels

@testset "Optimizations" begin
    @testset "optimize_kernel_local" begin
        T = 100
        y = randn(T)
        h_bw_values = [0.1, 0.5, 1.0]
        best = optimize_kernel_local(y; T_train = T - T÷2, 
                                        kernel_dict = Dict{Symbol, Function}(:epanechnikov => kernel_dict[:epanechnikov]),
                                        bandwidths = h_bw_values)
        @test best.h_bw in h_bw_values
        @test best.kernel in keys(kernel_dict)
    end

    @testset "optimize_kernel_regression" begin
        T = 100
        y = randn(T)
        X = randn(T, 3)
        h_bw_values = [0.1, 0.5, 1.0]
        best = optimize_kernel_regression(y, X; T_train = T - T÷2,
                                                 kernel_dict = Dict{Symbol, Function}(:epanechnikov => kernel_dict[:epanechnikov]),
                                                 bandwidths = h_bw_values)
        @test best.h_bw in h_bw_values
        @test best.kernel in keys(kernel_dict)
    end

    @testset "optimize_kernel_ar" begin
        T = 100
        y = randn(T)
        p = 2
        h_bw_values = [0.1, 0.5, 1.0]
        best = optimize_kernel_ar(y, p; T_train = T - T÷2,
                                       kernel_dict = Dict{Symbol, Function}(:epanechnikov => kernel_dict[:epanechnikov]),
                                       bandwidths = h_bw_values)
        @test best.h_bw in h_bw_values
        @test best.kernel in keys(kernel_dict)
    end

    @testset "optimize_lowess_regression" begin
        T = 100
        y = randn(T)
        X = randn(T, 3)
        h_bw_values = [0.1, 0.5, 1.0]
        best = optimize_lowess_regression(y, X; T_train = T - T÷2,
                                                 kernel_dict = Dict{Symbol, Function}(:epanechnikov => kernel_dict[:epanechnikov]),
                                                 bandwidths = h_bw_values)
        @test best.h_bw in h_bw_values
        @test best.kernel in keys(kernel_dict)
    end

    @testset "optimize_lowess_ar" begin
        T = 100
        y = randn(T)
        p = 2
        h_bw_values = [0.1, 0.5, 1.0]
        best = optimize_lowess_ar(y, p; T_train = T - T÷2,
                                       kernel_dict = Dict{Symbol, Function}(:epanechnikov => kernel_dict[:epanechnikov]),
                                       bandwidths = h_bw_values)
        @test best.h_bw in h_bw_values
        @test best.kernel in keys(kernel_dict)
    end

    @testset "optimize_QBLL_regression" begin
        T = 100
        y = randn(T)
        X = randn(T, 3)
        h_bw_values = [1, 5, 10]
        best = optimize_QBLL_regression(y, X; T_train = T - T÷2,
                                               kernel_dict = Dict{Symbol, Function}(:epanechnikov => kernel_dict[:epanechnikov]),
                                               bandwidths = h_bw_values,
                                               n_iter = 100)
        @test best.h_bw in h_bw_values
        @test best.kernel in keys(kernel_dict)
    end

    @testset "optimize_QBLL_ar" begin
        T = 100
        y = randn(T)
        p = 2
        h_bw_values = [1, 5, 10]
        best = optimize_QBLL_ar(y, p; T_train = T - T÷2,
                                     kernel_dict = Dict{Symbol, Function}(:epanechnikov => kernel_dict[:epanechnikov]),
                                     bandwidths = h_bw_values,
                                     n_iter = 100)
        @test best.h_bw in h_bw_values
        @test best.kernel in keys(kernel_dict)
    end
end