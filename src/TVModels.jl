module TVModels


include("kernels.jl")
include("rolling_regressions.jl")
include("simulate_series.jl")

# Export
export epanechnikov, uniform_kernel, triangular_kernel, gaussian_kernel,
       quartic_kernel, triweight_kernel, tricube_kernel, cosine_kernel,
       logistic_kernel, sigmoid_kernel, silverman_kernel, kernel_dict

export rolling_regression, rolling_ar

export simulate_tv_arp, simulate_ar


end
