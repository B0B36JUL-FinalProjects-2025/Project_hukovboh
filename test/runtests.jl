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
    end
end
