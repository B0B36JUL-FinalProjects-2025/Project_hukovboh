using LinearAlgebra

"""
    state_local_level(y::AbstractVector; sigma_eta::Real, sigma_eps::Real, a0::Real, P0::Real, smooth::Bool=true)

Fit a local level state space model using Kalman filtering and smoothing.

# Model
```
y_t = a_t + ϵ_t,  ϵ_t ~ N(0, σ²ₑₚₛ)
a_t = a_{t-1} + η_t,  η_t ~ N(0, σ²ₑₜₐ)
a_0 ~ N(a0, P0)
```

# Arguments
- `y::AbstractVector`: Univariate time series observations (must have at least 2 observations)
- `sigma_eta::Real`: Standard deviation of state noise (must be > 0)
- `sigma_eps::Real`: Standard deviation of observation noise (must be > 0)
- `a0::Real`: Initial state mean
- `P0::Real`: Initial state covariance (must be > 0)
- `smooth::Bool`: Whether to apply backward smoothing. (default: true)

# Returns
If `smooth=true`: tuple of (a_filt, P_filt, a_smooth, P_smooth). 
If `smooth=false`: tuple of (a_filt, P_filt)
where a_filt and P_filt are T-length vectors of filtered estimates,
and a_smooth and P_smooth are T-length vectors of smoothed estimates.
a_filt and P_filt utilize information up to time t (filtered estimates).
a_smooth and P_smooth utilize information from the entire series (smoothed estimates).
"""
function state_local_level(y::AbstractVector;
                            sigma_eta::Real,
                            sigma_eps::Real,
                            a0::Real,
                            P0::Real,
                            smooth::Bool=true)
    # y_t = a_t + ϵ_t,   ϵ_t ~ N(0, σ_eps²)
    # a_t = a_{t-1} + η_t,  η_t ~ N(0, σ_eta²)
    # a_0 ~ N(a0, P0)

    if P0 <= 0.0
        throw(DomainError("Initial state covariance P0 must be positive"))
    end
    if sigma_eta <= 0.0
        throw(DomainError("State noise standard deviation sigma_eta must be positive"))
    end
    if sigma_eps <= 0.0
        throw(DomainError("Observation noise standard deviation sigma_eps must be positive"))
    end
    if length(y) < 2
        throw(DomainError("Input time series y must have at least two observations"))
    end
    
    T = length(y)

    # State variable mean and covariance storage
    a_pred = zeros(Float64, T)   # a_{t|t-1}
    P_pred = zeros(Float64, T)
    a_filt = zeros(Float64, T)   # a_{t|t}
    P_filt = zeros(Float64, T)

    # Forward pass: filter. Begin with prior
    a_prev = a0
    P_prev = P0

    for t in 1:T
        # Prediction. Assume random walk of state variable a_t = a_{t-1} + η_t
        a_t = a_prev
        P_t = P_prev + sigma_eta^2

        # Update. y_t = a_t + ϵ_t
        y_pred = a_t
        v_t = y[t] - y_pred
        F_t = P_t + sigma_eps^2
        K_t = P_t / F_t

        # Filter
        a_upd = a_t + K_t * v_t
        P_upd = P_t - K_t^2*F_t

        a_pred[t] = a_t
        P_pred[t] = P_t
        a_filt[t] = a_upd
        P_filt[t] = P_upd

        a_prev = a_upd
        P_prev = P_upd
    end

    if smooth == true
        # Backward pass: smoother
        a_smooth = similar(a_filt)
        P_smooth = similar(P_filt)

        a_smooth[end] = a_filt[end]
        P_smooth[end] = P_filt[end]

        for t in (T-1):-1:1
            P_f = P_filt[t]
            P_p_next = P_pred[t+1]
            # Scalar smoother gain
            J_t = P_f / P_p_next         

            a_smooth[t] = a_filt[t] + J_t * (a_smooth[t+1] - a_pred[t+1])
            P_smooth[t] = P_f + J_t^2 * (P_smooth[t+1] - P_p_next)
        end

        return a_filt, P_filt, a_smooth, P_smooth
    else
        return a_filt, P_filt
    end
end

# helpers
_toP0(P0::AbstractMatrix, k) = P0
_toP0(P0::AbstractVector, k) = Diagonal(P0)
_toP0(P0::Real, k) = P0 .* I(k)

_toQ(Q::AbstractMatrix, k) = Q
_toQ(Q::AbstractVector, k) = Diagonal(Q)
_toQ(Q::Real, k) = Q .* I(k)

function _state_regression(y_target::AbstractVector, X::AbstractMatrix,
                        sigma_eps::Real,
                        Q::AbstractMatrix,
                        β0::AbstractVector,
                        P0::AbstractMatrix;
                        state_transition::Function = (β_filtered, P_filtered, Q) -> (β_filtered[end, :], P_filtered[end, :, :] + Q),
                        smooth::Bool=true)
    # y_t = β_0t + ∑β_kt*x_kt + ϵ_t,   ϵ_t ~ N(0, σ_eps²)
    # β_t = β_{t-1} + η_t,  η_t ~ N(0, Q)
    # β_0 ~ N(a0, P0)

    T_target = length(y_target)
    k = size(X, 2) + 1

    if T_target < k
        throw(DomainError("Not enough data points to estimate the model"))
    end
    if size(X, 1) != T_target
        throw(DimensionMismatch("Length of y must match number of rows in X"))
    end
    if length(β0) != k
        throw(DomainError("Initial state mean β0 must have length k (where k = number of features + 1)"))
    end
    if size(Q, 1) != k || size(Q, 2) != k
        throw(DomainError("State noise covariance Q must be of size k × k (where k = number of features + 1)"))
    end
    if size(P0, 1) != k || size(P0, 2) != k
        throw(DomainError("Initial state covariance P0 must be of size k × k (where k = number of features + 1)"))
    end
    if size(Q) != size(P0)
        throw(DomainError("State noise covariance Q and initial state covariance P0 must have the same dimensions"))
    end
    if sigma_eps <= 0.0
        throw(DomainError("Observation noise standard deviation sigma_eps must be positive"))
    end
    if any(eigvals(Q) .<= 0.0) || issymmetric(Q) == false
        throw(DomainError("State noise covariance Q must be positive definite symmetric matrix"))
    end
    if any(eigvals(P0) .<= 0.0) || issymmetric(P0) == false
        throw(DomainError("Initial state covariance P0 must be positive definite symmetric matrix"))
    end

    # Add bias to X
    X = hcat(ones(T_target), X)

    # Forward pass: filter. Begin with prior
    β_prev = β0
    P_prev = P0

    # State variable mean and covariance storage
    β_pred = fill(NaN, T_target, k)   # a_{t|t-1}
    P_pred = Array{Float64,3}(undef, T_target, k, k)
    β_filt = fill(NaN, T_target, k)   # a_{t|t}
    P_filt = Array{Float64,3}(undef, T_target, k, k)

    for t in 1:T_target
        # Prediction step using flexible state transition
        β_t, P_t = nothing, nothing
        try
            if t == 1
                β_t, P_t = state_transition(reshape(β_prev, 1, k), reshape(P_prev, 1, k, k), Q)
            else
                β_t, P_t = state_transition(β_filt[1:t-1, :], P_filt[1:t-1, :, :], Q)
            end
        catch e
            throw(ArgumentError("Error in state_transition function: ", e))
        end

        # Update. y_t = β_0t + ∑β_kt*x_kt + ϵ_t
        x_t = X[t, :]
        y_pred = dot(β_t, x_t)
        v_t = y_target[t] - y_pred
        F_t = x_t' * P_t * x_t + sigma_eps^2
        K_t = P_t * x_t / F_t

        # Filter
        β_upd = β_t + K_t * v_t
        P_upd = P_t - K_t * F_t * K_t'

        β_pred[t, :] = β_t
        P_pred[t, :, :] = P_t
        β_filt[t, :] = β_upd
        P_filt[t, :, :] = P_upd

        β_prev = β_upd
        P_prev = P_upd
    end

    if smooth == true
        # Backward pass: smoother
        β_smooth = similar(β_filt)
        P_smooth = similar(P_filt)

        β_smooth[end, :] = β_filt[end, :]
        P_smooth[end, :, :] = P_filt[end, :, :]

        for t in (T_target-1):-1:1
            P_f = P_filt[t, :, :]
            P_p_next = P_pred[t+1, :, :]
            # Scalar smoother gain
            J_t = P_f / P_p_next         

            β_smooth[t, :] = β_filt[t, :] + J_t * (β_smooth[t+1, :] - β_pred[t+1, :])
            P_smooth[t, :, :] = P_f + J_t * (P_smooth[t+1, :, :] - P_p_next) * J_t'
        end

        return β_filt, P_filt, β_smooth, P_smooth
    else
        return β_filt, P_filt
    end
end


"""
    state_regression(y_target::AbstractVector, X::Union{AbstractVector, AbstractMatrix};
                    sigma_eps::Real,
                    Q::Union{Real, AbstractVector, AbstractMatrix},
                    β0::AbstractVector,
                    P0::Union{Real, AbstractVector, AbstractMatrix},
                    state_transition::Function = (β_filtered, P_filtered, Q) -> (β_filtered[end, :], P_filtered[end, :, :] + Q),
                    smooth::Bool=true)


Fit a time-varying regression state space model using Kalman filtering and smoothing.

# Model with random walk state evolution
```
y_t = β_{0t} + ∑_{k=1}^d β_{kt} * x_{kt} + ϵ_t,  ϵ_t ~ N(0, σ²ₑₚₛ)
β_t = β_{t-1} + η_t,  η_t ~ N(0, Q)
β₀ ~ N(β0, P0)
```

# Arguments
- `y_target::AbstractVector`: Target time series observations
- `X::Union{AbstractVector, AbstractMatrix}`: Regressor matrix (or vector for univariate case)
- `sigma_eps::Real`: Standard deviation of observation noise (must be > 0)
- `Q`: State noise covariance matrix of size (d+1)×(d+1) (can be Real, Vector, or Matrix)
- `β0::AbstractVector`: Initial state mean (length must be d + 1)
- `P0`: Initial state covariance matrix of size (d+1)×(d+1) (can be Real, Vector, or Matrix)
- `state_transition::Function`: Function defining state evolution taking as inputs all previous 
    betas (t-1) x k), their previous covariances (t-1) x k X k), and Q, 
    defaults to random walk: (β_filtered, P_filtered, Q) -> (β_filtered[end, :], P_filtered[end, :, :] + Q)
- `smooth::Bool`: Whether to apply backward smoothing. (default: true)

# Returns
If `smooth=true`: tuple of (β_filt, P_filt, β_smooth, P_smooth).
If `smooth=false`: tuple of (β_filt, P_filt)
where each β is (T)×(d+1) matrix and P is (T)×(d+1)×(d+1) array of estimates.
β_filt and P_filt are utilizing information up to time t (filtered estimates).
β_smooth and P_smooth are utilizing information from the entire series (smoothed estimates).
"""
function state_regression(y_target::AbstractVector, X::Union{AbstractVector, AbstractMatrix};
                        sigma_eps::Real,
                        Q::Union{Real, AbstractVector, AbstractMatrix},
                        β0::AbstractVector,
                        P0::Union{Real, AbstractVector, AbstractMatrix},
                        state_transition::Function = (β_filtered, P_filtered, Q) -> (β_filtered[end, :], P_filtered[end, :, :] + Q),
                        smooth::Bool=true)
    X_mat = isa(X, AbstractVector) ? reshape(X, :, 1) : X
    k = size(X_mat, 2) + 1
    _state_regression(y_target, X_mat,
                    sigma_eps,
                    _toQ(Q, k),
                    β0,
                    _toP0(P0, k);
                    state_transition=state_transition,
                    smooth=smooth)
end

"""
    state_ar(y::AbstractVector, p::Int; sigma_eps::Real, Q, β0::AbstractVector, P0, 
             state_transition::Function = (β_filtered, P_filtered, Q) -> (β_filtered[end, :], P_filtered[end, :, :] + Q), 
             smooth::Bool=true)

Fit a time-varying autoregressive state space model using Kalman filtering and smoothing.

# Model with random walk state evolution
```
y_t = β₀ₜ + ∑_{k=1}^p β_{k,t}*y_{t-k} + ϵ_t,  ϵ_t ~ N(0, σ²ₑₚₛ)
β_t = β_{t-1} + η_t,  η_t ~ N(0, Q)
β₀ ~ N(β0, P0)
```

# Arguments
- `y::AbstractVector`: Univariate time series observations
- `p::Int`: Autoregressive order (must be ≥ 1)
- `sigma_eps::Real`: Standard deviation of observation noise (must be > 0)
- `Q`: State noise covariance matrix of size (p+1)×(p+1) (can be Real, Vector, or Matrix)
- `β0::AbstractVector`: Initial state mean (length must be p + 1)
- `P0`: Initial state covariance matrix of size (p+1)×(p+1) (can be Real, Vector, or Matrix)
- `state_transition::Function`: Function defining state evolution taking as inputs all previous 
    betas (t-1) x k), their previous covariances (t-1) x k X k), and Q, 
    defaults to random walk: (β_filtered, P_filtered, Q) -> (β_filtered[end, :], P_filtered[end, :, :] + Q)
- `smooth::Bool`: Whether to apply backward smoothing. (default: true)

# Returns
If `smooth=true`: tuple of (β_filt, P_filt, β_smooth, P_smooth). 
If `smooth=false`: tuple of (β_filt, P_filt)
where each β is (T-p)×(p+1) matrix and P is (T-p)×(p+1)×(p+1) array of estimates.
β_filt and P_filt are utilizing information up to time t (filtered estimates).
β_smooth and P_smooth are utilizing information from the entire series (smoothed estimates).
"""
function state_ar(y::AbstractVector, p::Int;
                  sigma_eps::Real,
                  Q::Union{Real, AbstractVector, AbstractMatrix},
                  β0::AbstractVector,
                  P0::Union{Real, AbstractVector, AbstractMatrix},
                  state_transition::Function = (β_filtered, P_filtered, Q) -> (β_filtered[end, :], P_filtered[end, :, :] + Q),
                  smooth::Bool=true)
    if p < 1
        throw(DomainError("Order p must be at least 1"))
    end

    y_target, X = _create_ar_variables(y, p)
    return state_regression(y_target, X; sigma_eps, Q, β0, P0, state_transition=state_transition, smooth=smooth)
end
