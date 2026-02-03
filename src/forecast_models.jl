
"""
    forecast_regression(X::AbstractMatrix, β::AbstractMatrix)

Compute regression one-ahead forecasts for multiple series using coefficient matrix.

# Arguments
- `X::AbstractMatrix`: Input features matrix
- `β::AbstractMatrix`: Coefficient matrix where first column is intercept and remaining columns are slopes

# Returns
- `AbstractMatrix`: Forecasted values
"""
function forecast_regression(X::AbstractMatrix, β::AbstractMatrix)
    if size(X, 2) != size(β, 2) - 1
        throw(DimensionMismatch("Number of columns in X must match number of coefficient slopes in β"))
    end
    if isempty(β) || isempty(X)
        throw(DomainError("Coefficient matrix β and input matrix X cannot be empty"))
    end
    if size(X, 1) != size(β, 1)
        throw(DimensionMismatch("Number of rows in X must match number of rows in β"))
    end

    c = β[:, 1]
    ϕ = β[:, 2:end]

    return c + sum(ϕ .* X, dims=2)
end

function forecast_regression(X::AbstractVector, β::AbstractVector)
    if length(X) != length(β) - 1
        throw(DimensionMismatch("Length of X must match number of coefficient slopes in β"))
    end
    if isempty(β)
        throw(DomainError("Coefficient vector β cannot be empty"))
    end

    c = β[1]
    ϕ = β[2:end]

    return c + sum(ϕ .* X)
end

"""
    forecast_ar(X::AbstractVector, β::AbstractVector, h::Int)

Compute multi-step ahead forecasts for an autoregressive model.

# Arguments
- `X::AbstractVector`: Lags of the time series used as predictors
- `β::AbstractVector`: Autoregressive coefficients where first element is intercept and remaining elements are AR lags
- `h::Int`: Number of forecast steps ahead

# Returns
- `AbstractVector`: Forecasted values for h steps ahead
"""
function forecast_ar(X::AbstractVector, β::AbstractVector, h::Int)
    if length(X) < length(β) - 1
        throw(DimensionMismatch("Length of X must be at least number of coefficient slopes in β"))
    end
    if isempty(β)
        throw(DomainError("Coefficient vector β cannot be empty"))
    end
    if h < 1
        throw(DomainError("Number of forecast steps h must be at least 1"))
    end

    y_forecast = Vector{Float64}(undef, h)
    p = length(β) - 1
    X_extended = vcat(X, zeros(h))

    for t in 1:h
        y_forecast[t] = β[1] + sum(β[2:end] .* reverse(X_extended[(length(X) + t - p):(length(X) + t - 1)]))
        X_extended[length(X) + t] = y_forecast[t]
    end

    return y_forecast
end

"""
    forecast_state_ar(X::AbstractVector, β_filt::AbstractMatrix, P_filt::AbstractMatrix, h::Int;
                      Q::AbstractMatrix,
                      state_transition::Function = (β_filtered, P_filtered, Q) -> (β_filtered[end, :], P_filtered[end, :, :] + Q))

Compute multi-step ahead forecasts for a state-space autoregressive model with time-varying coefficients.

# Arguments
- `X::AbstractVector`: Lags of the time series used as predictors
- `β_filt::AbstractMatrix`: Filtered coefficient matrix where first column is intercept and remaining columns are AR lags
- `P_filt::AbstractMatrix`: Filtered covariance matrices of coefficients
- `h::Int`: Number of forecast steps ahead
- `Q::AbstractMatrix`: State noise covariance matrix
- `state_transition::Function`: Function defining state evolution taking as inputs all previous 
    betas (t-1) x k), their previous covariances (t-1) x k X k), and Q, 
    defaults to random walk: (β_filtered, P_filtered, Q) -> (β_filtered[end, :], P_filtered[end, :, :] + Q)

# Returns
- `AbstractVector`: Forecasted values for h steps ahead
"""
function forecast_state_ar(X::AbstractVector, β_filt::AbstractMatrix, P_filt::AbstractMatrix, h::Int;
                        Q::AbstractMatrix,
                        state_transition::Function = (β_filtered, P_filtered, Q) -> (β_filtered[end, :], P_filtered[end, :, :] + Q))
    if length(X) < size(β_filt, 2) - 1
        throw(DimensionMismatch("Length of X must be at least number of coefficient slopes in β_filt"))
    end
    if isempty(β_filt)
        throw(DomainError("Coefficient matrix β_filt cannot be empty"))
    end
    if size(β_filt, 1) != size(P_filt, 1)
        throw(DimensionMismatch("Number of rows in β_filt must match number of covariance matrices in P_filt"))
    end
    if size(β_filt, 2) != size(P_filt, 2)
        throw(DimensionMismatch("Number of columns in β_filt must match size of covariance matrices in P_filt"))
    end
    if size(P_filt, 2) != size(P_filt, 3)
        throw(DimensionMismatch("Covariance matrices in P_filt must be square"))
    end
    if h < 1
        throw(DomainError("Number of forecast steps h must be at least 1"))
    end
    if size(Q, 1) != size(Q, 2)
        throw(DimensionMismatch("State noise covariance matrix Q must be square"))
    end
    if any(eigvals(Q) .<= 0.0) || issymmetric(Q) == false
        throw(DomainError("State noise covariance Q must be positive definite symmetric matrix"))
    end
    
    p = size(β_filt, 2) - 1
    y_forecast = Vector{Float64}(undef, h)
    X_extended = vcat(X, zeros(h))

    for t in 1:h
        β_pred, P_pred = state_transition(β_filt, P_filt, Q)
        y_forecast[t] = β_pred[1] + sum(β_pred[2:end] .* reverse(X_extended[(length(X) + t - p):(length(X) + t - 1)]))
        X_extended[length(X) + t] = y_forecast[t]
        β_filt = vcat(β_filt, reshape(β_pred, 1, :))
        P_filt = cat(P_filt, reshape(P_pred, 1, size(P_pred, 1), size(P_pred, 2)); dims=1)
    end

    return y_forecast
end

"""
    forecast_local_level(a::Real, h::Int)

Compute constant level forecasts for h steps ahead. Aplicable for local-level state-space models and kernel local-level and local-linear models.

# Arguments
- `a::Real`: Last predicted level
- `h::Int`: Number of forecast steps ahead

# Returns
- `AbstractVector`: Forecasted values (constant level repeated h times)
"""
function forecast_local_level(a::Real, h::Int)
    if h < 1
        throw(DomainError("Number of forecast steps h must be at least 1"))
    end

    return fill(a, h)
end