module TVModels


include("kernels.jl")

# Export
export epanechnikov, uniform_kernel, triangular_kernel, gaussian_kernel,
       quartic_kernel, triweight_kernel, tricube_kernel, cosine_kernel,
       logistic_kernel, sigmoid_kernel, silverman_kernel, kernel_dict



end
