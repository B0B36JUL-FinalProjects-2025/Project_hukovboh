
using Distributions


"""
    epanechnikov(u::Real) -> Float64

Epanechnikov kernel function.

# Arguments
- `u::Real`: Standardized distance from the target point

# Returns
- `Float64`: Kernel value, 0.75 * (1 - u²) for |u| ≤ 1, and 0.0 otherwise

# Example
```julia
epanechnikov(0.5)   # Returns 0.5625
epanechnikov(1.5)   # Returns 0.0
```
"""
epanechnikov(u::Real) = abs(u) <= 1 ? 0.75 * (1 - u^2) : 0.0

"""
    uniform_kernel(u::Real) -> Float64

Uniform kernel function.

# Arguments
- `u::Real`: Standardized distance from the target point

# Returns
- `Float64`: Kernel value, 0.5 for |u| ≤ 1, and 0.0 otherwise

# Example
```julia
uniform_kernel(0.5)   # Returns 0.5
uniform_kernel(1.5)   # Returns 0.0
```
"""
uniform_kernel(u::Real) = abs(u) <= 1 ? 0.5 : 0.0

"""
    triangular_kernel(u::Real) -> Float64

Triangular kernel function.

# Arguments
- `u::Real`: Standardized distance from the target point

# Returns
- `Float64`: Kernel value, (1 - |u|) for |u| ≤ 1, and 0.0 otherwise

# Example
```julia
triangular_kernel(0.5)   # Returns 0.5
triangular_kernel(1.5)   # Returns 0.0
```
"""
triangular_kernel(u::Real) = abs(u) <= 1 ? (1 - abs(u)) : 0.0

"""
    gaussian_kernel(u::Real) -> Float64

Gaussian kernel function.

# Arguments
- `u::Real`: Standardized distance from the target point

# Returns
- `Float64`: Kernel value, probability density function of standard normal distribution

# Example
```julia
gaussian_kernel(0.0)   # Returns ≈ 0.3989
gaussian_kernel(1.0)   # Returns ≈ 0.2419
```
"""
gaussian_kernel(u::Real) = pdf(Normal(0, 1), u)

"""
    quartic_kernel(u::Real) -> Float64

Quartic kernel function.

# Arguments
- `u::Real`: Standardized distance from the target point

# Returns
- `Float64`: Kernel value, (15/16) * (1 - u²)² for |u| ≤ 1, and 0.0 otherwise

# Example
```julia
quartic_kernel(0.5)   # Returns 0.3164...
quartic_kernel(1.5)   # Returns 0.0
```
"""
quartic_kernel(u::Real) = abs(u) <= 1 ? (15/16) * (1 - u^2)^2 : 0.0

"""
    triweight_kernel(u::Real) -> Float64

Triweight kernel function.

# Arguments
- `u::Real`: Standardized distance from the target point

# Returns
- `Float64`: Kernel value, (35/32) * (1 - u²)³ for |u| ≤ 1, and 0.0 otherwise

# Example
```julia
triweight_kernel(0.5)   # Returns ≈ 0.2246
triweight_kernel(1.5)   # Returns 0.0
```
"""
triweight_kernel(u::Real) = abs(u) <= 1 ? (35/32) * (1 - u^2)^3 : 0.0

"""
    tricube_kernel(u::Real) -> Float64

Tricube kernel function.

# Arguments
- `u::Real`: Standardized distance from the target point

# Returns
- `Float64`: Kernel value, (70/81) * (1 - |u|³)³ for |u| ≤ 1, and 0.0 otherwise

# Example
```julia
tricube_kernel(0.5)   # Returns ≈ 0.2330
tricube_kernel(1.5)   # Returns 0.0
```
"""
tricube_kernel(u::Real) = abs(u) <= 1 ? (70/81) * (1 - abs(u)^3)^3 : 0.0

"""
    cosine_kernel(u::Real) -> Float64

Cosine kernel function.

# Arguments
- `u::Real`: Standardized distance from the target point

# Returns
- `Float64`: Kernel value, (π/4) * cos((π/2) * u) for |u| ≤ 1, and 0.0 otherwise

# Example
```julia
cosine_kernel(0.0)   # Returns ≈ 0.7854
cosine_kernel(1.5)   # Returns 0.0
```
"""
cosine_kernel(u::Real) = abs(u) <= 1 ? (π/4) * cos((π/2) * u) : 0.0

"""
    logistic_kernel(u::Real) -> Float64

Logistic kernel function.

# Arguments
- `u::Real`: Standardized distance from the target point

# Returns
- `Float64`: Kernel value, 1 / (exp(u) + 2 + exp(-u))

# Example
```julia
logistic_kernel(0.0)   # Returns 0.25
logistic_kernel(1.0)   # Returns ≈ 0.0820
```
"""
logistic_kernel(u::Real) = 1 / (exp(u) + 2 + exp(-u))

"""
    sigmoid_kernel(u::Real) -> Float64

Sigmoid kernel function.

# Arguments
- `u::Real`: Standardized distance from the target point

# Returns
- `Float64`: Kernel value, (2/π) * (1 / (exp(u) + exp(-u)))

# Example
```julia
sigmoid_kernel(0.0)   # Returns ≈ 0.6366
sigmoid_kernel(1.0)   # Returns ≈ 0.2084
```
"""
sigmoid_kernel(u::Real) = (2 / π) * (1 / (exp(u) + exp(-u)))

"""
    silverman_kernel(u::Real) -> Float64

Silverman kernel function.

# Arguments
- `u::Real`: Standardized distance from the target point

# Returns
- `Float64`: Kernel value, 1/2 * exp(-|u|/√2) * sin(|u|/√2 + π/4)

# Example
```julia
silverman_kernel(0.0)   # Returns ≈ 0.3827
silverman_kernel(1.0)   # Returns ≈ 0.1065
silverman_kernel(5.0)   # Returns ≈ -0.0134
```
"""
silverman_kernel(u::Real) = 1/2 * exp(-abs(u)/sqrt(2)) * sin(abs(u)/sqrt(2) + π/4)

# Dictionary of available kernels
const kernel_dict = Dict(
    :epanechnikov   => epanechnikov,
    :uniform        => uniform_kernel,
    :triangular     => triangular_kernel,
    :gaussian       => gaussian_kernel,
    :quartic        => quartic_kernel,
    :triweight      => triweight_kernel,
    :tricube        => tricube_kernel,
    :cosine         => cosine_kernel,
    :logistic       => logistic_kernel,
    :sigmoid        => sigmoid_kernel,
    :silverman      => silverman_kernel
)
