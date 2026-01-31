using TVModels
using Test
using Aqua

@testset "TVModels.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(TVModels)
    end
    # Write your tests here.
end
