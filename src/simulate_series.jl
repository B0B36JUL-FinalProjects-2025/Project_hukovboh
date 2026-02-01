
using Random

"""
    simulate_tv_ar(T::Int, βs::AbstractMatrix; σ::Real=1.0, seed::Union{Int, Nothing}=nothing)

Simulate a time-varying autoregressive process with time-dependent coefficients provided in matrix form.

# Arguments
- `T::Int`: Length of the time series to simulate
- `βs::AbstractMatrix`: Matrix of coefficients of size (T, p+1) where p is the order of the AR process (first column is intercept)
- `σ::Real=1.0`: Standard deviation of the error term (must be non-negative)
- `seed::Union{Int, Nothing}=nothing`: Random seed for reproducibility

# Returns
- `Vector`: Simulated time series of length T
"""
function simulate_tv_ar(T::Int, βs::AbstractMatrix; σ::Real=1.0, seed::Union{Int, Nothing}=nothing)
    p  = size(βs, 2) - 1

    if seed !== nothing
        Random.seed!(seed)
    end
    if isempty(βs)
        throw(DomainError("βs matrix cannot be empty"))
    end
    if T < p + 1
        throw(DomainError("Length T must be at least p + 1"))
    end
    if σ < 0.0
        throw(DomainError("Standard deviation σ must be non-negative"))
    end
    if size(βs, 1) != T
        throw(DimensionMismatch("βs must be of size (T, p+1)"))
    end

    y = zeros(T+p)
    ϵ = randn(T)
    for t in (p+1):(T+p)
        y[t] = βs[t-p, 1] + reverse(βs[t-p, 2:end])' * y[(t-p):(t-1)] + σ * ϵ[t-p]
    end

    return y[(p+1):end]
end

"""
    simulate_tv_ar(T::Int, β_funcs::AbstractVector{<:Function}; σ::Real=1.0, seed::Union{Int, Nothing}=nothing)

Simulate a time-varying autoregressive process with time-dependent coefficients provided as functions.

# Arguments
- `T::Int`: Length of the time series to simulate
- `β_funcs::AbstractVector{<:Function}`: Vector of p functions, each taking a time value in [0, 1] and returning a coefficient. The functions will be evaluated at T equally spaced points in [0, 1].
- `σ::Real=1.0`: Standard deviation of the error term (must be non-negative)
- `seed::Union{Int, Nothing}=nothing`: Random seed for reproducibility

# Returns
- `Vector`: Simulated time series of length T
"""
function simulate_tv_ar(T::Int, β_funcs::AbstractVector{<:Function}; σ::Real=1.0, seed::Union{Int, Nothing}=nothing)
    p = length(β_funcs)
    βs = zeros(T, p)
    ts = range(0, stop=1, length=T)
    for i in 1:p
        try
            βs[:, i] = β_funcs[i].(ts)
        catch e
            throw(ArgumentError("Function at index $i failed to evaluate: $(typeof(e)) - $e"))
        end
    end
    return simulate_tv_ar(T, βs; σ=σ, seed=seed)
end

"""
    simulate_ar(T::Int, βs::AbstractVector; σ::Real=1.0, seed::Union{Int, Nothing}=nothing)

Simulate a standard autoregressive process with constant coefficients

# Arguments
- `T::Int`: Length of the time series to simulate
- `βs::AbstractVector`: Vector of coefficients of length p+1 for an AR(p) process (first element is intercept)
- `σ::Real=1.0`: Standard deviation of the error term (must be non-negative)
- `seed::Union{Int, Nothing}=nothing`: Random seed for reproducibility

# Returns
- `Vector`: Simulated time series of length T
"""
function simulate_ar(T::Int, βs::AbstractVector; σ::Real=1.0, seed::Union{Int, Nothing}=nothing)
    p = length(βs) - 1

    if seed !== nothing
        Random.seed!(seed)
    end
    if isempty(βs)
        throw(DomainError("βs vector cannot be empty"))
    end
    if T < p + 1
        throw(DomainError("Length T must be at least p + 1"))
    end
    if σ < 0.0
        throw(DomainError("Standard deviation σ must be non-negative"))
    end

    y = zeros(T+p)
    ϵ = randn(T)
    for t in (p+1):(T+p)
        y[t] = βs[1] + reverse(βs[2:end])' * y[(t-p):(t-1)] + σ * ϵ[t-p]
    end

    return y[(p+1):end]
end