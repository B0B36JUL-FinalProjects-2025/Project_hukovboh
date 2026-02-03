

"""
    optimize_kernel_local(y::AbstractVector;
                         T_train::Int,
                         kernel_dict::Dict{Symbol,Function} = kernel_dict,
                         bandwidths::Vector{Float64} = [0.001,0.01,0.02,0.05,0.08,0.10,0.15,0.20,0.25,0.50,1.0],
                         criterion::Function = (y_true, y_pred) -> mean((y_true .- y_pred).^2),
                         horizon::Int = 1,
                         local_linear::Bool = false,
                         return_forecasts::Bool = false)

Optimize kernel parameters for local level or local linear forecasting models.

# Arguments
- `y::AbstractVector`: Time series data to forecast
- `T_train::Int`: Number of observations to use for training
- `kernel_dict::Dict{Symbol,Function}`: Dictionary mapping kernel names to kernel functions
- `bandwidths::Vector{Float64}`: Bandwidth values to test
- `criterion::Function`: Loss function for model evaluation (default: MSE)
- `horizon::Int`: Forecasting horizon (default: 1)
- `local_linear::Bool`: If true, use local linear; if false, use local level (default: false)
- `return_forecasts::Bool`: If true, return both best parameters and forecasts (default: false)

# Returns
- If `return_forecasts=false`: Best model configuration as a NamedTuple
- If `return_forecasts=true`: Tuple of (best_config, (y_pred, y_true))

# Examples
```julia
best = optimize_kernel_local(y, T_train=100)
best, (y_pred, y_true) = optimize_kernel_local(y, T_train=100, return_forecasts=true)
```
"""
function optimize_kernel_local(y::AbstractVector;
                            T_train::Int,
                            kernel_dict::Dict{Symbol,Function} = kernel_dict,
                            bandwidths::Vector{Float64} = [0.001,0.01,0.02,0.05,0.08,0.10,0.15,0.20,0.25,0.50,1.0],
                            criterion::Function = (y_true, y_pred) -> mean((y_true .- y_pred).^2),
                            horizon::Int = 1,
                            local_linear::Bool = false,
                            return_forecasts::Bool = false)


    T = length(y)

    if T_train < 1 || T_train >= T
        error("T_train must be between 1 and length(y)-1.")
    end

    T_test = T - T_train
    y_pred = fill(NaN, T_test - horizon + 1)
    y_true = y[(T_train + horizon):end]

    results = NamedTuple[]

    for (kname, kern) in kernel_dict, h_bw in bandwidths
        for (j, t) in enumerate((T_train + 1):(T-horizon+1))
            # history up to t-1 only
            y_hist = y[1:(t-1)]
            try
                if local_linear
                    at = kernel_local_linear(y_hist, h_bw; kernel = kern)
                else
                    at = kernel_local_level(y_hist, h_bw; kernel = kern)
                end
                y_pred[j] = forecast_local_level(at[end], horizon)[end]
            catch e
                @warn "Error fitting kernel model for kernel=$(kname), h_bw=$(h_bw) at time t=$(t): $e"
                continue
            end
        end

        try
            push!(results, (
                kernel = kname,
                h_bw   = h_bw,
                criterion    = criterion(y_true, y_pred)
            ))
        catch e
            @warn "Error computing criterion for kernel=$(kname), h_bw=$(h_bw): $e"
        end
    end

    # Select best configuration
    results = sort(results, by = r -> r.criterion)
    best = results[1]

    # Estimate the best model
    for (j, t) in enumerate((T_train + 1):(T-horizon+1))
        # history up to t-1 only
        y_hist = y[1:(t-1)]
        if local_linear
            at = kernel_local_linear(y_hist, best.h_bw; kernel = kernel_dict[best.kernel])
        else
            at = kernel_local_level(y_hist, best.h_bw; kernel = kernel_dict[best.kernel])
        end
        y_pred[j] = forecast_local_level(at[end], horizon)[end]
    end

    if return_forecasts
        return best, (y_pred, y_true)
    else
        return best
    end
end

"""
    optimize_kernel_regression(y::AbstractVector, X::AbstractMatrix;
                            T_train::Int,
                            kernel_dict::Dict{Symbol,Function} = kernel_dict,
                            bandwidths::Vector{Float64} = [0.001,0.01,0.02,0.05,0.08,0.10,0.15,0.20,0.25,0.50,1.0],
                            criterion::Function = (y_true, y_pred) -> mean((y_true .- y_pred).^2),
                            return_forecasts::Bool = false)

Optimize kernel parameters for kernel regression forecasting models.

# Arguments
- `y::AbstractVector`: Time series data to forecast
- `X::AbstractMatrix`: Exogenous variables (features) for regression
- `T_train::Int`: Number of observations to use for training
- `kernel_dict::Dict{Symbol,Function}`: Dictionary mapping kernel names to kernel functions
- `bandwidths::Vector{Float64}`: Bandwidth values to test
- `criterion::Function`: Loss function for model evaluation (default: MSE)
- `return_forecasts::Bool`: If true, return both best parameters and forecasts (default: false)

# Returns
- If `return_forecasts=false`: Best model configuration as a NamedTuple
- If `return_forecasts=true`: Tuple of (best_config, (y_pred, y_true))

# Examples
```julia
best = optimize_kernel_regression(y, X, T_train=100)
best, (y_pred, y_true) = optimize_kernel_regression(y, X, T_train=100, return_forecasts=true)
```
"""
function optimize_kernel_regression(y::AbstractVector, X::AbstractMatrix;
                            T_train::Int,
                            kernel_dict::Dict{Symbol,Function} = kernel_dict,
                            bandwidths::Vector{Float64} = [0.001,0.01,0.02,0.05,0.08,0.10,0.15,0.20,0.25,0.50,1.0],
                            criterion::Function = (y_true, y_pred) -> mean((y_true .- y_pred).^2),
                            return_forecasts::Bool = false)

    T = length(y)

    if T_train < 1 || T_train >= T
        error("T_train must be between 1 and length(y)-1.")
    end

    T_test = T - T_train
    y_pred = fill(NaN, T_test)
    y_true = y[(T_train + 1):end]

    results = NamedTuple[]

    for (kname, kern) in kernel_dict, h_bw in bandwidths
        for (j, t) in enumerate((T_train + 1):T)
            # history up to t-1 only
            y_hist = y[1:(t-1)]
            X_hist = X[1:(t-1), :]
            
            try
                betas = kernel_regression(y_hist, X_hist, h_bw; kernel = kern)
                y_pred[j] = forecast_regression(X[t, :], betas[end, :])
            catch e
                @warn "Error fitting kernel regression for kernel=$(kname), h_bw=$(h_bw) at time t=$(t): $e"
                continue
            end
        end
        try
            push!(results, (
                kernel = kname,
                h_bw   = h_bw,
                criterion    = criterion(y_true, y_pred)
            ))
        catch e
            @warn "Error computing criterion for kernel=$(kname), h_bw=$(h_bw): $e"
        end
    end

    # Select best configuration
    results = sort(results, by = r -> r.criterion)
    best = results[1]

    # Estimate the best model
    for (j, t) in enumerate((T_train + 1):T)
        # history up to t-1 only
        y_hist = y[1:(t-1)]
        X_hist = X[1:(t-1), :]

        betas = kernel_regression(y_hist, X_hist, best.h_bw; kernel = kernel_dict[best.kernel])
        y_pred[j] = forecast_regression(X[t, :], betas[end, :])
    end

    if return_forecasts
        return best, (y_pred, y_true)
    else
        return best
    end
end

"""
    optimize_kernel_ar(y::AbstractVector, p::Int;
                       T_train::Int,
                       kernel_dict::Dict{Symbol,Function} = kernel_dict,
                       bandwidths::Vector{Float64} = [0.001,0.01,0.02,0.05,0.08,0.10,0.15,0.20,0.25,0.50,1.0],
                       criterion::Function = (y_true, y_pred) -> mean((y_true .- y_pred).^2),
                       horizon::Int = 1,
                       return_forecasts::Bool = false)

Optimize kernel parameters for kernel autoregressive forecasting models.

# Arguments
- `y::AbstractVector`: Time series data to forecast
- `p::Int`: Autoregressive order
- `T_train::Int`: Number of observations to use for training
- `kernel_dict::Dict{Symbol,Function}`: Dictionary mapping kernel names to kernel functions
- `bandwidths::Vector{Float64}`: Bandwidth values to test
- `criterion::Function`: Loss function for model evaluation (default: MSE)
- `horizon::Int`: Forecasting horizon (default: 1)
- `return_forecasts::Bool`: If true, return both best parameters and forecasts (default: false)

# Returns
- If `return_forecasts=false`: Best model configuration as a NamedTuple
- If `return_forecasts=true`: Tuple of (best_config, (y_pred, y_true))

# Examples
```julia
best = optimize_kernel_ar(y, 1, T_train=100)
best, (y_pred, y_true) = optimize_kernel_ar(y, 1, T_train=100, return_forecasts=true)
```
"""
function optimize_kernel_ar(y::AbstractVector, p::Int;
                            T_train::Int,
                            kernel_dict::Dict{Symbol,Function} = kernel_dict,
                            bandwidths::Vector{Float64} = [0.001,0.01,0.02,0.05,0.08,0.10,0.15,0.20,0.25,0.50,1.0],
                            criterion::Function = (y_true, y_pred) -> mean((y_true .- y_pred).^2),
                            horizon::Int = 1,
                            return_forecasts::Bool = false)

    T = length(y)

    if T_train < 1 || T_train >= T
        error("T_train must be between 1 and length(y)-1.")
    end

    T_test = T - T_train
    y_pred = fill(NaN, T_test - horizon + 1)
    y_true = y[(T_train + horizon):end]

    results = NamedTuple[]

    for (kname, kern) in kernel_dict, h_bw in bandwidths
        for (j, t) in enumerate((T_train + 1):(T-horizon+1))
            # history up to t-1 only
            y_hist = y[1:(t-1)]

            try
                betas = kernel_ar(y_hist, p, h_bw; kernel = kern)
                y_pred[j] = forecast_ar(y_hist[end-p+1:end], betas[end, :], horizon)[end]
            catch e
                @warn "Error fitting kernel AR model for kernel=$(kname), h_bw=$(h_bw) at time t=$(t): $e"
                continue
            end
        end

        try
            push!(results, (
                kernel = kname,
                h_bw   = h_bw,
                criterion    = criterion(y_true, y_pred)
            ))
        catch e
            @warn "Error computing criterion for kernel=$(kname), h_bw=$(h_bw): $e"
        end
    end

    # Select best configuration
    results = sort(results, by = r -> r.criterion)
    best = results[1]

    # Estimate the best model
    for (j, t) in enumerate((T_train + 1):(T-horizon+1))
        # history up to t-1 only
        y_hist = y[1:(t-1)]
        betas = kernel_ar(y_hist, p, best.h_bw; kernel = kernel_dict[best.kernel])
        y_pred[j] = forecast_ar(y_hist[end-p+1:end], betas[end, :], horizon)[end]
    end

    if return_forecasts
        return best, (y_pred, y_true)
    else
        return best
    end
end

"""
    optimize_lowess_regression(y::AbstractVector, X::AbstractMatrix;
                              T_train::Int,
                              kernel_dict::Dict{Symbol,Function} = kernel_dict,
                              bandwidths::Vector{Float64} = [0.001,0.01,0.02,0.05,0.08,0.10,0.15,0.20,0.25,0.50,1.0],
                              n_iter::Int = 3,
                              criterion::Function = (y_true, y_pred) -> mean((y_true .- y_pred).^2),
                              return_forecasts::Bool = false)

Optimize kernel parameters for LOWESS (Locally Weighted Scatterplot Smoothing) regression forecasting models.

# Arguments
- `y::AbstractVector`: Time series data to forecast
- `X::AbstractMatrix`: Exogenous variables (features) for regression
- `T_train::Int`: Number of observations to use for training
- `kernel_dict::Dict{Symbol,Function}`: Dictionary mapping kernel names to kernel functions
- `bandwidths::Vector{Float64}`: Bandwidth values to test
- `n_iter::Int`: Number of LOWESS iterations (default: 3)
- `criterion::Function`: Loss function for model evaluation (default: MSE)
- `return_forecasts::Bool`: If true, return both best parameters and forecasts (default: false)

# Returns
- If `return_forecasts=false`: Best model configuration as a NamedTuple
- If `return_forecasts=true`: Tuple of (best_config, (y_pred, y_true))

# Examples
```julia
best = optimize_lowess_regression(y, X, T_train=100)
best, (y_pred, y_true) = optimize_lowess_regression(y, X, T_train=100, return_forecasts=true)
```
"""
function optimize_lowess_regression(y::AbstractVector, X::AbstractMatrix;
                                    T_train::Int,
                                    kernel_dict::Dict{Symbol,Function} = kernel_dict,
                                    bandwidths::Vector{Float64} = [0.001,0.01,0.02,0.05,0.08,0.10,0.15,0.20,0.25,0.50,1.0],
                                    n_iter::Int = 3,
                                    criterion::Function = (y_true, y_pred) -> mean((y_true .- y_pred).^2),
                                    return_forecasts::Bool = false)

    T = length(y)

    if T_train < 1 || T_train >= T
        error("T_train must be between 1 and length(y)-1.")
    end

    T_test = T - T_train
    y_pred = fill(NaN, T_test)
    y_true = y[(T_train + 1):end]

    results = NamedTuple[]

    for (kname, kern) in kernel_dict, h_bw in bandwidths
        for (j, t) in enumerate((T_train + 1):T)
            # history up to t-1 only
            y_hist = y[1:(t-1)]
            X_hist = X[1:(t-1), :]

            try
                betas = lowess_regression(y_hist, X_hist, h_bw, n_iter; kernel = kern)
                y_pred[j] = forecast_regression(X[t, :], betas[end, :])
            catch e
                @warn "Error fitting LOWESS regression for kernel=$(kname), h_bw=$(h_bw) at time t=$(t): $e"
                continue
            end
        end

        try
            push!(results, (
                kernel = kname,
                h_bw   = h_bw,
                criterion    = criterion(y_true, y_pred)
            ))
        catch e
            @warn "Error computing criterion for kernel=$(kname), h_bw=$(h_bw): $e"
        end
    end

    # Select best configuration
    results = sort(results, by = r -> r.criterion)
    best = results[1]

    # Estimate the best model
    for (j, t) in enumerate((T_train + 1):T)
        # history up to t-1 only
        y_hist = y[1:(t-1)]
        X_hist = X[1:(t-1), :]

        betas = lowess_regression(y_hist, X_hist, best.h_bw, n_iter; kernel = kernel_dict[best.kernel])
        y_pred[j] = forecast_regression(X[t, :], betas[end, :])
    end

    if return_forecasts
        return best, (y_pred, y_true)
    else
        return best
    end
end

"""
    optimize_lowess_ar(y::AbstractVector, p::Int;
                       T_train::Int,
                       kernel_dict::Dict{Symbol,Function} = kernel_dict,
                       bandwidths::Vector{Float64} = [0.001,0.01,0.02,0.05,0.08,0.10,0.15,0.20,0.25,0.50,1.0],
                       n_iter::Int = 3,
                       criterion::Function = (y_true, y_pred) -> mean((y_true .- y_pred).^2),
                       horizon::Int = 1,
                       return_forecasts::Bool = false)

Optimize kernel parameters for LOWESS (Locally Weighted Scatterplot Smoothing) autoregressive forecasting models.

# Arguments
- `y::AbstractVector`: Time series data to forecast
- `p::Int`: Autoregressive order
- `T_train::Int`: Number of observations to use for training
- `kernel_dict::Dict{Symbol,Function}`: Dictionary mapping kernel names to kernel functions
- `bandwidths::Vector{Float64}`: Bandwidth values to test
- `n_iter::Int`: Number of LOWESS iterations (default: 3)
- `criterion::Function`: Loss function for model evaluation (default: MSE)
- `horizon::Int`: Forecasting horizon (default: 1)
- `return_forecasts::Bool`: If true, return both best parameters and forecasts (default: false)

# Returns
- If `return_forecasts=false`: Best model configuration as a NamedTuple
- If `return_forecasts=true`: Tuple of (best_config, (y_pred, y_true))

# Examples
```julia
best = optimize_lowess_ar(y, 1, T_train=100)
best, (y_pred, y_true) = optimize_lowess_ar(y, 1, T_train=100, return_forecasts=true)
```
"""
function optimize_lowess_ar(y::AbstractVector, p::Int;
                            T_train::Int,
                            kernel_dict::Dict{Symbol,Function} = kernel_dict,
                            bandwidths::Vector{Float64} = [0.001,0.01,0.02,0.05,0.08,0.10,0.15,0.20,0.25,0.50,1.0],
                            n_iter::Int = 3,
                            criterion::Function = (y_true, y_pred) -> mean((y_true .- y_pred).^2),
                            horizon::Int = 1,
                            return_forecasts::Bool = false)

    T = length(y)

    if T_train < 1 || T_train >= T
        error("T_train must be between 1 and length(y)-1.")
    end

    T_test = T - T_train
    y_pred = fill(NaN, T_test - horizon + 1)
    y_true = y[(T_train + horizon):end]

    results = NamedTuple[]

    for (kname, kern) in kernel_dict, h_bw in bandwidths
        for (j, t) in enumerate((T_train + 1):(T-horizon+1))
            # history up to t-1 only
            y_hist = y[1:(t-1)]

            try
                betas = lowess_ar(y_hist, p, h_bw, n_iter; kernel = kern)
                y_pred[j] = forecast_ar(y_hist[end-p+1:end], betas[end, :], horizon)[end]
            catch e
                @warn "Error fitting LOWESS AR model for kernel=$(kname), h_bw=$(h_bw) at time t=$(t): $e"
                continue
            end
        end

        try
            push!(results, (
                kernel = kname,
                h_bw   = h_bw,
                criterion    = criterion(y_true, y_pred)
            ))
        catch e
            @warn "Error computing criterion for kernel=$(kname), h_bw=$(h_bw): $e"
        end
    end

    # Select best configuration
    results = sort(results, by = r -> r.criterion)
    best = results[1]

    # Estimate the best model
    for (j, t) in enumerate((T_train + 1):(T-horizon+1))
        # history up to t-1 only
        y_hist = y[1:(t-1)]
        betas = lowess_ar(y_hist, p, best.h_bw, n_iter; kernel = kernel_dict[best.kernel])
        y_pred[j] = forecast_ar(y_hist[end-p+1:end], betas[end, :], horizon)[end]
    end

    if return_forecasts
        return best, (y_pred, y_true)
    else
        return best
    end
end

"""
    optimize_QBLL_regression(y::AbstractVector, X::AbstractMatrix;
                            T_train::Int,
                            kernel_dict::Dict{Symbol,Function} = kernel_dict,
                            bandwidths::Vector{Float64} = [1,2,5,10,15,20,25,50,100],
                            n_iter::Int = 1000,
                            criterion::Function = (y_true, y_pred) -> mean((y_true .- y_pred).^2),
                            return_forecasts::Bool = false)

Optimize kernel parameters for QBLL (Quasi-Bayesian Local Level) regression forecasting models.

# Arguments
- `y::AbstractVector`: Time series data to forecast
- `X::AbstractMatrix`: Exogenous variables (features) for regression
- `T_train::Int`: Number of observations to use for training
- `kernel_dict::Dict{Symbol,Function}`: Dictionary mapping kernel names to kernel functions
- `bandwidths::Vector{Float64}`: Bandwidth values to test (default: [1,2,5,10,15,20,25,50,100])
- `n_iter::Int`: Number of QBLL iterations (default: 1000)
- `criterion::Function`: Loss function for model evaluation (default: MSE)
- `return_forecasts::Bool`: If true, return both best parameters and forecasts (default: false)

# Returns
- If `return_forecasts=false`: Best model configuration as a NamedTuple
- If `return_forecasts=true`: Tuple of (best_config, (y_pred, y_true))

# Examples
```julia
best = optimize_QBLL_regression(y, X, T_train=100)
best, (y_pred, y_true) = optimize_QBLL_regression(y, X, T_train=100, return_forecasts=true)
```
"""
function optimize_QBLL_regression(y::AbstractVector, X::AbstractMatrix;
                            T_train::Int,
                            kernel_dict::Dict{Symbol,Function} = kernel_dict,
                            bandwidths::Vector{<:Real} = [1,2,5,10,15,20,25,50,100],
                            n_iter::Int = 1000,
                            criterion::Function = (y_true, y_pred) -> mean((y_true .- y_pred).^2),
                            return_forecasts::Bool = false)

    T = length(y)

    if T_train < 1 || T_train >= T
        error("T_train must be between 1 and length(y)-1.")
    end

    T_test = T - T_train
    y_pred = fill(NaN, T_test)
    y_true = y[(T_train + 1):end]

    results = NamedTuple[]

    for (kname, kern) in kernel_dict, h_bw in bandwidths
        for (j, t) in enumerate((T_train + 1):T)
            # history up to t-1 only
            y_hist = y[1:(t-1)]
            X_hist = X[1:(t-1), :]

            try
                betas = QBLL_regression(y_hist, X_hist, n_iter, h_bw; kernel = kern)
                y_pred[j] = forecast_regression(X[t, :], betas[end, :])
            catch e
                @warn "Error fitting QBLL regression for kernel=$(kname), h_bw=$(h_bw) at time t=$(t): $e"
                continue
            end
        end

        try
            push!(results, (
                kernel = kname,
                h_bw   = h_bw,
                criterion    = criterion(y_true, y_pred)
            ))
        catch e
            @warn "Error computing criterion for kernel=$(kname), h_bw=$(h_bw): $e"
        end
    end

    # Select best configuration
    results = sort(results, by = r -> r.criterion)
    best = results[1]

    # Estimate the best model
    for (j, t) in enumerate((T_train + 1):T)
        # history up to t-1 only
        y_hist = y[1:(t-1)]
        X_hist = X[1:(t-1), :]

        betas = QBLL_regression(y_hist, X_hist, n_iter, best.h_bw; kernel = kernel_dict[best.kernel])
        y_pred[j] = forecast_regression(X[t, :], betas[end, :])
    end

    if return_forecasts
        return best, (y_pred, y_true)
    else
        return best
    end
end

"""
    optimize_QBLL_ar(y::AbstractVector, p::Int;
                     T_train::Int,
                     kernel_dict::Dict{Symbol,Function} = kernel_dict,
                     bandwidths::Vector{Float64} = [1,2,5,10,15,20,25,50,100],
                     n_iter::Int = 1000,
                     criterion::Function = (y_true, y_pred) -> mean((y_true .- y_pred).^2),
                     horizon::Int = 1,
                     return_forecasts::Bool = false)

Optimize kernel parameters for QBLL (Quasi-Bayesian Local Level) autoregressive forecasting models.

# Arguments
- `y::AbstractVector`: Time series data to forecast
- `p::Int`: Autoregressive order
- `T_train::Int`: Number of observations to use for training
- `kernel_dict::Dict{Symbol,Function}`: Dictionary mapping kernel names to kernel functions
- `bandwidths::Vector{Float64}`: Bandwidth values to test (default: [1,2,5,10,15,20,25,50,100])
- `n_iter::Int`: Number of QBLL iterations (default: 1000)
- `criterion::Function`: Loss function for model evaluation (default: MSE)
- `horizon::Int`: Forecasting horizon (default: 1)
- `return_forecasts::Bool`: If true, return both best parameters and forecasts (default: false)

# Returns
- If `return_forecasts=false`: Best model configuration as a NamedTuple
- If `return_forecasts=true`: Tuple of (best_config, (y_pred, y_true))

# Examples
```julia
best = optimize_QBLL_ar(y, 1, T_train=100)
best, (y_pred, y_true) = optimize_QBLL_ar(y, 1, T_train=100, return_forecasts=true)
```
"""
function optimize_QBLL_ar(y::AbstractVector, p::Int;
                            T_train::Int,
                            kernel_dict::Dict{Symbol,Function} = kernel_dict,
                            bandwidths::Vector{<:Real} = [1,2,5,10,15,20,25,50,100],
                            n_iter::Int = 1000,
                            criterion::Function = (y_true, y_pred) -> mean((y_true .- y_pred).^2),
                            horizon::Int = 1,
                            return_forecasts::Bool = false)

    T = length(y)

    if T_train < 1 || T_train >= T
        error("T_train must be between 1 and length(y)-1.")
    end

    T_test = T - T_train
    y_pred = fill(NaN, T_test - horizon + 1)
    y_true = y[(T_train + horizon):end]

    results = NamedTuple[]

    for (kname, kern) in kernel_dict, h_bw in bandwidths
        for (j, t) in enumerate((T_train + 1):(T-horizon+1))
            # history up to t-1 only
            y_hist = y[1:(t-1)]

            try
                betas = QBLL_ar(y_hist, p, n_iter, h_bw; kernel = kern)
                y_pred[j] = forecast_ar(y_hist[end-p+1:end], betas[end, :], horizon)[end]
            catch e
                @warn "Error fitting QBLL AR model for kernel=$(kname), h_bw=$(h_bw) at time t=$(t): $e"
                continue
            end
        end

        try
            push!(results, (
                kernel = kname,
                h_bw   = h_bw,
                criterion    = criterion(y_true, y_pred)
            ))
        catch e
            @warn "Error computing criterion for kernel=$(kname), h_bw=$(h_bw): $e"
        end
    end

    # Select best configuration
    results = sort(results, by = r -> r.criterion)
    best = results[1]

    # Estimate the best model
    for (j, t) in enumerate((T_train + 1):(T-horizon+1))
        # history up to t-1 only
        y_hist = y[1:(t-1)]
        betas = QBLL_ar(y_hist, p, n_iter, best.h_bw; kernel = kernel_dict[best.kernel])
        y_pred[j] = forecast_ar(y_hist[end-p+1:end], betas[end, :], horizon)[end]
    end

    if return_forecasts
        return best, (y_pred, y_true)
    else
        return best
    end
end