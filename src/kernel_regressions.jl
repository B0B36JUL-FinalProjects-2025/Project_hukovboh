
using LinearAlgebra

"""
    kernel_local_level(y::AbstractVector, bandwidth::Float64; kernel::Function=epanechnikov_kernel, min_n::Int=8)

Performs time-varying level estimation using kernel local-level regression.

# Arguments
- `y::AbstractVector`: Time series vector
- `bandwidth::Float64`: Kernel bandwidth parameter in (0, 1]
- `kernel::Function`: Kernel function to use (default: epanechnikov_kernel). Can be any user-defined function that takes a Real input and returns a Float64 output.
- `min_n::Int`: Minimum number of observations with positive weights for parameter estimation (default: 8). If not enough NaNs are returned for that time point.

# Returns
- `levels::Vector`: Vector of shape (T,) containing time-varying level estimates for each time point t = 1, ..., T
"""
function kernel_local_level(y::AbstractVector, bandwidth::Float64; 
                            kernel::Function=epanechnikov_kernel,
                            min_n::Int=8)

    T = length(y)

    # Validate inputs
    if T < 1
        throw(DomainError("Not enough data points to estimate the model"))
    end
    if bandwidth <= 0.0 || bandwidth > 1.0
        throw(DomainError("Bandwidth must be in (0, 1]"))
    end

    ty = collect(1:T) ./ T
    levels = fill(NaN, T)

    for idx_t in 1:T  
        t0 = ty[idx_t]
        dt = ty .- t0

        # Create weights for kernel regression
        w = nothing
        try
            w = kernel.(dt ./ bandwidth) ./ bandwidth
        catch e
            throw(ArgumentError("Kernel function failed to evaluate: $(typeof(e)) - $e"))
        end
        
        # Positive weights only
        idx = findall(w .> 0)
        # Skip if not enough points
        if length(idx) < min_n
            continue
        end

        yloc = y[idx]
        wroot = sqrt.(w[idx])
        ytilde = yloc .* wroot

        level_estimate = sum(ytilde .* wroot) / sum(w[idx])
        levels[idx_t] = level_estimate
    end

    return levels
end

"""
    kernel_local_linear(y::AbstractVector, bandwidth::Float64; kernel::Function=epanechnikov_kernel, min_n::Int=8)

Performs time-varying level estimation using kernel local-linear regression.

# Arguments
- `y::AbstractVector`: Time series vector
- `bandwidth::Float64`: Kernel bandwidth parameter in (0, 1]
- `kernel::Function`: Kernel function to use (default: epanechnikov_kernel). Can be any user-defined function that takes a Real input and returns a Float64 output.
- `min_n::Int`: Minimum number of observations with positive weights for parameter estimation (default: 8). If not enough NaNs are returned for that time point.

# Returns
- `levels::Vector`: Vector of shape (T,) containing time-varying level estimates for each time point t = 1, ..., T
"""
function kernel_local_linear(y::AbstractVector, bandwidth::Float64; 
                            kernel::Function=epanechnikov_kernel,
                            min_n::Int=8)

    T = length(y)

    # Validate inputs
    if T < 1
        throw(DomainError("Not enough data points to estimate the model"))
    end
    if bandwidth <= 0.0 || bandwidth > 1.0
        throw(DomainError("Bandwidth must be in (0, 1]"))
    end

    ty = collect(1:T) ./ T
    levels = fill(NaN, T)

    for idx_t in 1:T  
        t0 = ty[idx_t]
        dt = ty .- t0

        # Create weights for kernel regression
        w = nothing
        try
            w = kernel.(dt ./ bandwidth) ./ bandwidth
        catch e
            throw(ArgumentError("Kernel function failed to evaluate: $(typeof(e)) - $e"))
        end
        
        # Positive weights only
        idx = findall(w .> 0)
        # Skip if not enough points
        if length(idx) < min_n
            continue
        end

        # Construct local-linear design
        nloc = length(idx)
        Z = ones(nloc, 2)
        Z[:, 2] .= dt[idx]

        yloc = y[idx]
        wroot = sqrt.(w[idx])
        Ztilde = Z .* wroot
        ytilde = yloc .* wroot

        θ = Ztilde \ ytilde
        # Local-linear estimate at u0
        level_estimate = θ[1]

        levels[idx_t] = level_estimate
    end

    return levels
end

"""
    kernel_ar(y::AbstractVector, p::Int, bandwidth::Float64; kernel::Function=epanechnikov_kernel, min_n::Int=8)

Performs time-varying autoregressive (AR) estimation with local-linear kernel estimation.

# Arguments
- `y::AbstractVector`: Time series vector
- `p::Int`: Order of the autoregressive model (must be ≥ 1)
- `bandwidth::Float64`: Kernel bandwidth parameter in (0, 1]
- `kernel::Function`: Kernel function to use (default: epanechnikov_kernel). Can be any user-defined function that takes a Real input and returns a Float64 output.
- `min_n::Int`: Minimum number of observations with positive weights for parameter estimation (default: 8). If not enough NaNs are returned for that time point.

# Returns
- `betas::Matrix`: Matrix of shape (T-p, p+1) containing time-varying AR coefficients (intercept + lag coefficients) for each time point t = p+1, ..., T
"""
function kernel_ar(y::AbstractVector, p::Int, bandwidth::Float64; 
                    kernel::Function=epanechnikov_kernel,
                    min_n::Int=8)

    T = length(y)
    k = p + 1

    # Validate inputs
    if T < p + 1
        throw(DomainError("Not enough data points to estimate the model"))
    end
    if p < 1
        throw(DomainError("Order p must be at least 1"))
    end
    if bandwidth <= 0.0 || bandwidth > 1.0
        throw(DomainError("Bandwidth must be in (0, 1]"))
    end

    ts = collect(1:T) ./ T
    betas = fill(NaN, T-p, k)

    # Construct lagged design matrix without intercept
    X = ones(T - p, p)
    for j in 1:p
        X[:, j] = y[(p - j + 1):(end - j)]
    end
    y_target = y[(p + 1):end]
    ty = ts[(p + 1):end]

    # Corresponds to t = p+idx_t
    for idx_t in 1:length(y_target)  
        t0 = ty[idx_t]
        dt = ty .- t0

        # Create weights for kernel regression
        w = nothing
        try
            w = kernel.(dt ./ bandwidth) ./ bandwidth
        catch e
            throw(ArgumentError("Kernel function failed to evaluate: $(typeof(e)) - $e"))
        end
        
        # Positive weights only
        idx = findall(w .> 0)
        # Skip if not enough points
        if length(idx) < min_n
            continue
        end

        # Construct local-linear design:
        # y = a0 + Σ b_j X_j + a1 du + Σ d_j du*X_j
        # => regressors length = 2k
        nloc = length(idx)
        Z = ones(nloc, 2k)

        # Level terms (β0..βp)
        for j in 1:p
            Z[:, j+1] .= X[idx, j]
        end

        # Intercept slope
        Z[:, k+1] .= dt[idx]
        # Slope terms in dt
        for j in 1:p
            Z[:, k+1+j] .= dt[idx] .* X[idx, j]
        end

        yloc = y_target[idx]
        wroot = sqrt.(w[idx])
        Ztilde = Z .* wroot
        ytilde = yloc .* wroot

        θ = Ztilde \ ytilde
        # Local-linear estimates at u0
        betas_loc = θ[1:k]    

        betas[idx_t, :] .= betas_loc
    end

    return betas
end


"""
    kernel_regression(y_target::AbstractVector, X::AbstractMatrix, bandwidth::Float64; kernel::Function=epanechnikov_kernel, min_n::Int=8)

Performs time-varying kernel regression with local-linear kernel estimation.

# Arguments
- `y_target::AbstractVector`: Target variable vector
- `X::AbstractMatrix`: Design matrix of shape (T, p) containing regressors
- `bandwidth::Float64`: Kernel bandwidth parameter in (0, 1]
- `kernel::Function`: Kernel function to use (default: epanechnikov_kernel). Can be any user-defined function that takes a Real input and returns a Float64 output.
- `min_n::Int`: Minimum number of observations with positive weights for parameter estimation (default: 8). If not enough NaNs are returned for that time point.

# Returns
- `betas::Matrix`: Matrix of shape (T, p+1) containing time-varying regression coefficients (intercept + slope coefficients) for each time point t = 1, ..., T
"""
function kernel_regression(y_target::AbstractVector, X::AbstractMatrix, bandwidth::Float64; 
                        kernel::Function=epanechnikov_kernel,
                        min_n::Int=8)

    T = length(y_target)
    k = size(X, 2) + 1

    # Validate inputs
    if T < k
        throw(DomainError("Not enough data points to estimate the model"))
    end
    if size(X, 1) != T
        throw(DimensionMismatch("Length of y must match number of rows in X"))
    end
    if bandwidth <= 0.0 || bandwidth > 1.0
        throw(DomainError("Bandwidth must be in (0, 1]"))
    end

    ty = collect(1:T) ./ T
    betas = fill(NaN, T, k)

    # Corresponds to t = p+idx_t
    for idx_t in 1:length(y_target)  
        t0 = ty[idx_t]
        dt = ty .- t0

        # Create weights for kernel regression
        w = nothing
        try
            w = kernel.(dt ./ bandwidth) ./ bandwidth
        catch e
            throw(ArgumentError("Kernel function failed to evaluate: $(typeof(e)) - $e"))
        end
        
        # Positive weights only
        idx = findall(w .> 0)
        # Skip if not enough points
        if length(idx) < min_n
            continue
        end

        # Construct local-linear design:
        # y = a0 + Σ b_j X_j + a1 du + Σ d_j du*X_j
        # => regressors length = 2k
        nloc = length(idx)
        Z = ones(nloc, 2k)

        # Level terms (β0..βp)
        for j in 1:k-1
            Z[:, j+1] .= X[idx, j]
        end

        # Intercept slope
        Z[:, k+1] .= dt[idx]
        # Slope terms in dt
        for j in 1:k-1
            Z[:, k+1+j] .= dt[idx] .* X[idx, j]
        end

        yloc = y_target[idx]
        wroot = sqrt.(w[idx])
        Ztilde = Z .* wroot
        ytilde = yloc .* wroot

        θ = Ztilde \ ytilde
        # Local-linear estimates at u0
        betas_loc = θ[1:k]    

        betas[idx_t, :] .= betas_loc
    end

    return betas
end

"""
    kernel_regression(y_target::AbstractVector, X::AbstractVector, bandwidth::Float64; kernel::Function=epanechnikov_kernel, min_n::Int=8)

Performs time-varying kernel regression with local-linear kernel estimation.

# Arguments
- `y_target::AbstractVector`: Target variable vector
- `X::AbstractVector`: Vector of a single regressor
- `bandwidth::Float64`: Kernel bandwidth parameter in (0, 1]
- `kernel::Function`: Kernel function to use (default: epanechnikov_kernel). Can be any user-defined function that takes a Real input and returns a Float64 output.
- `min_n::Int`: Minimum number of observations with positive weights for parameter estimation (default: 8). If not enough NaNs are returned for that time point.

# Returns
- `betas::Matrix`: Matrix of shape (T, 2) containing time-varying regression coefficients (intercept + slope coefficient) for each time point t = 1, ..., T
"""
function kernel_regression(y_target::AbstractVector, X::AbstractVector, bandwidth::Float64; 
                        kernel::Function=epanechnikov_kernel,
                        min_n::Int=8)
    return kernel_regression(y_target, reshape(X, :, 1), bandwidth; 
                            kernel=kernel, min_n=min_n)
end

"""
    lowess_regression(y_target::AbstractVector, X::AbstractVector, bandwidth::Float64, n_iter::Int; 
                        kernel::Function=epanechnikov_kernel,
                        delta_kernel::Function=quartic_kernel,
                        min_n::Int=8)
Perform lowess (locally weighted scatterplot smoothing) regression with iterative reweighting.

# Arguments
- `y_target::AbstractVector`: Target variable vector
- `X::AbstractMatrix`: Matrix of regressors of shape (T, p)
- `bandwidth::Float64`: Kernel bandwidth parameter in (0, 1]
- `n_iter::Int`: Number of iterations for reweighting (must be at least 1)
- `kernel::Function`: Kernel function to use (default: epanechnikov_kernel). Can be any user-defined function that takes a Real input and returns a Float64 output.
- `delta_kernel::Function`: Delta kernel function for reweighting (default: quartic_kernel). Can be any user-defined function that takes a Real input and returns a Float64 output.
- `min_n::Int`: Minimum number of observations with positive weights for parameter estimation (default: 8). If not enough NaNs are returned for that time point.

# Returns
- `betas::Matrix`: Matrix of shape (T, p+1) containing time-varying regression coefficients (intercept + slope coefficients) for each time point t = 1, ..., T
"""
function lowess_regression(y_target::AbstractVector, X::AbstractMatrix, bandwidth::Float64, n_iter::Int; 
                        kernel::Function=epanechnikov_kernel,
                        delta_kernel::Function=quartic_kernel,
                        min_n::Int=8)
    T = length(y_target)
    k = size(X, 2) + 1

    # Validate inputs
    if T < k
        throw(DomainError("Not enough data points to estimate the model"))
    end
    if size(X, 1) != T
        throw(DimensionMismatch("Length of y must match number of rows in X"))
    end
    if bandwidth <= 0.0 || bandwidth > 1.0
        throw(DomainError("Bandwidth must be in (0, 1]"))
    end
    if n_iter < 1
        throw(DomainError("Number of iterations n_iter must be at least 1"))
    end

    ty = collect(1:T) ./ T
    betas = fill(NaN, T, k)

    # Lowess deltas and residuals
    δs = ones(T)
    residuals = similar(y_target)

    for iter in 1:n_iter
        # Corresponds to t = p+idx_t
        for idx_t in 1:length(y_target)  
            t0 = ty[idx_t]
            dt = ty .- t0

            # Create weights for kernel regression
            w = nothing
            try
                w = δs .* kernel.(dt ./ bandwidth) ./ bandwidth
            catch e
                throw(ArgumentError("Kernel function failed to evaluate: $(typeof(e)) - $e"))
            end
            
            # Positive weights only
            idx = findall(w .> 0)
            # Skip if not enough points
            if length(idx) < min_n
                continue
            end

            # Construct local-linear design:
            # y = a0 + Σ b_j X_j + a1 du + Σ d_j du*X_j
            # => regressors length = 2k
            nloc = length(idx)
            Z = ones(nloc, 2k)

            # Level terms (β0..βp)
            for j in 1:k-1
                Z[:, j+1] .= X[idx, j]
            end

            # Intercept slope
            Z[:, k+1] .= dt[idx]
            # Slope terms in dt
            for j in 1:k-1
                Z[:, k+1+j] .= dt[idx] .* X[idx, j]
            end

            yloc = y_target[idx]
            wroot = sqrt.(w[idx])
            Ztilde = Z .* wroot
            ytilde = yloc .* wroot

            θ = Ztilde \ ytilde
            # Local-linear estimates at u0
            betas_loc = θ[1:k]    

            betas[idx_t, :] .= betas_loc

            # Update residuals 
            residuals[idx_t] = y_target[idx_t] - dot(betas_loc, [1.0; X[idx_t, :]])
        end

        # Update δs based on residuals
        δs = delta_kernel.(residuals ./ (6 * median(abs.(residuals))))
    end

    return betas
end

"""
    lowess_regression(y_target::AbstractVector, X::AbstractVector, bandwidth::Float64, n_iter::Int; 
                        kernel::Function=epanechnikov_kernel,
                        delta_kernel::Function=quartic_kernel,
                        min_n::Int=8)

Perform lowess (locally weighted scatterplot smoothing) regression with iterative reweighting for a single regressor.

# Arguments
- `y_target::AbstractVector`: Target variable vector
- `X::AbstractVector`: Vector of a single regressor
- `bandwidth::Float64`: Kernel bandwidth parameter in (0, 1]
- `n_iter::Int`: Number of iterations for reweighting (must be at least 1)
- `kernel::Function`: Kernel function to use (default: epanechnikov_kernel). Can be any user-defined function that takes a Real input and returns a Float64 output.
- `delta_kernel::Function`: Delta kernel function for reweighting (default: quartic_kernel). Can be any user-defined function that takes a Real input and returns a Float64 output.
- `min_n::Int`: Minimum number of observations with positive weights for parameter estimation (default: 8). If not enough, NaNs are returned for that time point.

# Returns
- `betas::Matrix`: Matrix of shape (T, 2) containing time-varying regression coefficients (intercept + slope coefficient) for each time point t = 1, ..., T
"""
function lowess_regression(y_target::AbstractVector, X::AbstractVector, bandwidth::Float64, n_iter::Int; 
                        kernel::Function=epanechnikov_kernel,
                        delta_kernel::Function=quartic_kernel,
                        min_n::Int=8)
    return lowess_regression(y_target, reshape(X, :, 1), bandwidth, n_iter; 
                            kernel=kernel, delta_kernel=delta_kernel, min_n=min_n)
end


"""
    lowess_ar(y::AbstractVector, p::Int, bandwidth::Float64, n_iter::Int; 
              kernel::Function=epanechnikov_kernel,
              delta_kernel::Function=quartic_kernel,
              min_n::Int=8)

Performs time-varying autoregressive (AR) estimation with local-linear kernel estimation and LOWESS (locally weighted scatterplot smoothing) iterative reweighting.

# Arguments
- `y::AbstractVector`: Time series vector
- `p::Int`: Order of the autoregressive model (must be ≥ 1)
- `bandwidth::Float64`: Kernel bandwidth parameter in (0, 1]
- `n_iter::Int`: Number of iterations for reweighting (must be at least 1)
- `kernel::Function`: Kernel function to use (default: epanechnikov_kernel). Can be any user-defined function that takes a Real input and returns a Float64 output.
- `delta_kernel::Function`: Delta kernel function for reweighting (default: quartic_kernel). Can be any user-defined function that takes a Real input and returns a Float64 output.
- `min_n::Int`: Minimum number of observations with positive weights for parameter estimation (default: 8). If not enough, NaNs are returned for that time point.

# Returns
- `betas::Matrix`: Matrix of shape (T-p, p+1) containing time-varying AR coefficients (intercept + lag coefficients) for each time point t = p+1, ..., T
"""
function lowess_ar(y::AbstractVector, p::Int, bandwidth::Float64, n_iter::Int; 
                    kernel::Function=epanechnikov_kernel,
                    delta_kernel::Function=quartic_kernel,
                    min_n::Int=8)

    T = length(y)
    k = p + 1

    # Validate inputs
    if T < p + 1
        throw(DomainError("Not enough data points to estimate the model"))
    end
    if p < 1
        throw(DomainError("Order p must be at least 1"))
    end
    if bandwidth <= 0.0 || bandwidth > 1.0
        throw(DomainError("Bandwidth must be in (0, 1]"))
    end
    if n_iter < 1
        throw(DomainError("Number of iterations n_iter must be at least 1"))
    end

    ts = collect(1:T) ./ T
    betas = fill(NaN, T-p, k)

    # Construct lagged design matrix without intercept
    n_obs = T - p
    X = ones(n_obs, p)
    for j in 1:p
        X[:, j] = y[(p - j + 1):(end - j)]
    end
    y_target = y[(p + 1):end]
    ty = ts[(p + 1):end]

    # Lowess deltas and residuals
    δs = ones(length(y_target))
    residuals = similar(y_target)

    for iter in 1:n_iter
        # Corresponds to t = p+idx_t
        for idx_t in 1:length(y_target)  
            t0 = ty[idx_t]
            dt = ty .- t0

            # Create weights for kernel regression
            w = nothing
            try
                w = δs .* kernel.(dt ./ bandwidth) ./ bandwidth
            catch e
                throw(ArgumentError("Kernel function failed to evaluate: $(typeof(e)) - $e"))
            end
            
            # Positive weights only
            idx = findall(w .> 0)
            # Skip if not enough points
            if length(idx) < min_n
                continue
            end

            # Construct local-linear design:
            # y = a0 + Σ b_j X_j + a1 du + Σ d_j du*X_j
            # => regressors length = 2k
            nloc = length(idx)
            Z = ones(nloc, 2k)

            # Level terms (β0..βp)
            for j in 1:p
                Z[:, j+1] .= X[idx, j]
            end

            # Intercept slope
            Z[:, k+1] .= dt[idx]
            # Slope terms in dt
            for j in 1:p
                Z[:, k+1+j] .= dt[idx] .* X[idx, j]
            end

            yloc = y_target[idx]
            wroot = sqrt.(w[idx])
            Ztilde = Z .* wroot
            ytilde = yloc .* wroot

            θ = Ztilde \ ytilde
            # Local-linear estimates at u0
            betas_loc = θ[1:k]    

            betas[idx_t, :] .= betas_loc

            # Update residuals
            residuals[idx_t] = y_target[idx_t] - dot(betas_loc, [1.0; X[idx_t, :]])
        end

        # Update δs based on residuals
        δs = delta_kernel.(residuals ./ (6 * median(abs.(residuals))))
    end

    return betas
end