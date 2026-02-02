
"""
    rolling_regression(y::AbstractVector, X::AbstractMatrix, window_size::Int; expanding::Bool=false)

Perform rolling window regression on the response vector `y` and feature matrix `X`.

# Arguments
- `y::AbstractVector`: Response vector of length T
- `X::AbstractMatrix`: Feature matrix of size T × n_features
- `window_size::Int`: Size of the rolling window
- `expanding::Bool=false`: If true, use expanding window with training data for the first estimation corresponding to the window size; otherwise use fixed rolling window

# Returns
- `betas::Matrix{Float64}`: Matrix of shape n_windows × (n_features + 1) containing estimated coefficients (intercept + feature coefficients)
"""
function rolling_regression(y::AbstractVector, X::AbstractMatrix, window_size::Int; expanding::Bool=false)
    T = length(y)
    n_features = size(X, 2)

    if T < n_features + 1
        throw(DomainError("Not enough data points to estimate the model"))
    end
    if size(X, 1) != T
        throw(DimensionMismatch("Length of y must match number of rows in X"))
    end
    if window_size > T
        throw(DomainError("Window size cannot be larger than the length of the time series"))
    end
    if window_size < n_features + 1
        throw(DomainError("Window size must be at least number of features + 1 to estimate coefficients"))
    end

    n_windows = T - window_size + 1

    betas = Array{Float64}(undef, n_windows, n_features + 1)

    for t in 1:n_windows
        if expanding
            y_window = y[1:(t + window_size - 1)]
            X_window = X[1:(t + window_size - 1), :]
        else
            y_window = y[t:(t + window_size - 1)]
            X_window = X[t:(t + window_size - 1), :]
        end

        # Add intercept
        X_design = hcat(ones(size(X_window, 1)), X_window)

        # Estimate coefficients
        betas[t, :] = X_design \ y_window
    end

    return betas
end

"""
    rolling_regression(y::AbstractVector, X::AbstractVector, window_size::Int; expanding::Bool=false)

Perform rolling window regression on the response vector `y` and feature vector `X`.

# Arguments
- `y::AbstractVector`: Response vector of length T
- `X::AbstractVector`: Feature vector of length T
- `window_size::Int`: Size of the rolling window
- `expanding::Bool=false`: If true, use expanding window with training data for the first estimation corresponding to the window size; otherwise use fixed rolling window

# Returns
- `betas::Matrix{Float64}`: Matrix of shape n_windows × 2 containing estimated coefficients (intercept + feature coefficient)
"""
function rolling_regression(y::AbstractVector, X::AbstractVector, window_size::Int; expanding::Bool=false)
    return rolling_regression(y, reshape(X, :, 1), window_size; expanding=expanding)
end


"""
    rolling_ar(y::AbstractVector, p::Int, window_size::Int; expanding::Bool=false)

Perform rolling window autoregressive (AR) regression on the time series `y`.

# Arguments
- `y::AbstractVector`: Time series vector of length T
- `p::Int`: Order of the autoregressive model (number of lags)
- `window_size::Int`: Size of the rolling window
- `expanding::Bool=false`: If true, use expanding window with training data for the first estimation corresponding to the window size; otherwise use fixed rolling window

# Returns
- `betas::Matrix{Float64}`: Matrix of shape n_windows × (p + 1) containing estimated coefficients (intercept + lag coefficients)
"""
function rolling_ar(y::AbstractVector, p::Int, window_size::Int; expanding::Bool=false)
    T = length(y)
    # We lose p observations for lagging
    n_windows = T - window_size - p + 1

    if T < p + 1
        throw(DomainError("Not enough data points to estimate the model"))
    end
    if window_size > T
        throw(DomainError("Window size cannot be larger than the length of the time series"))
    end
    if p < 1
        throw(DomainError("Order p must be at least 1"))
    end
    if window_size < p + 1
        throw(DomainError("Window size must be at least number of features + 1 to estimate coefficients"))
    end

    betas = Array{Float64}(undef, n_windows, p + 1)  

    # Construct lagged design matrix
    X = ones(T - p, p + 1)
    for j in 1:p
        X[:, j + 1] = y[(p - j + 1):(end - j)]
    end
    y_target = y[(p + 1):end]

    for t in 1:n_windows
        if expanding
            y_window = y_target[1:(t + window_size - 1)]
            X_window = X[1:(t + window_size - 1), :]
        else
            y_window = y_target[t:(t + window_size - 1)]
            X_window = X[t:(t + window_size - 1), :]
        end

        # Estimate coefficients
        betas[t, :] = X_window \ y_window
    end

    return betas
end