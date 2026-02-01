using TVModels
using Test
using Aqua

@testset "TVModels.jl" begin
    # Quality
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(TVModels)
    end

    # Tests
    @testset "Tests" begin
        include("test_kernels.jl")
        include("test_simulations.jl")
        include("test_rolling_regressions.jl")
        include("test_kernel_regressions.jl")
    end
end
