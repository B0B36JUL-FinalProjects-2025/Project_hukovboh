module TVModels


include("kernels.jl")
include("rolling_regressions.jl")
include("kernel_regressions.jl")
include("simulate_series.jl")

# Export
export epanechnikov_kernel, uniform_kernel, triangular_kernel, gaussian_kernel,
       quartic_kernel, triweight_kernel, tricube_kernel, cosine_kernel,
       logistic_kernel, sigmoid_kernel, silverman_kernel, kernel_dict

export rolling_regression, rolling_ar

export kernel_regression, kernel_ar, lowess_ar, lowess_regression

export simulate_tv_ar, simulate_ar


end
