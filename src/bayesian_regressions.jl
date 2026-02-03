
using LinearAlgebra
using Distributions


function kernel_QBLL(T::Int, H::Real, kernel::Function)
    # Weights over each obs for each obs
    ww = zeros(T, T)
    for j in 1:T
        for i in 1:T
            ww[i,j] = kernel((i - j) / H)
        end     
    end 

    # Normalize weights
    s = sum(ww, dims=2)
    adjw = zeros(T, T)  

    for k in 1:T
    adjw[k, :] = ww[k, :] / s[k]
    end

    # Adjust weights for QBLL
    cons = sum(adjw .^ 2.0, dims = 2)

    for k in 1:T
        adjw[k, :] = (1.0 / cons[k]) * (adjw[k, :])
    end

    return adjw
end

"""
    QBLL_regression(y::AbstractVector, X::AbstractMatrix, n_iter::Int, H::Real; kernel::Function=epanechnikov_kernel)

Performs Quasi-Bayesian Local Likelihood (QBLL) regression.

Arguments:
    y::AbstractVector: Response variable vector.
    X::AbstractMatrix: Predictor variables matrix.
    n_iter::Int: Number of posterior simulations per time point.
    H::Real: Bandwidth parameter for kernel weighting from interval [1, T] where T is the number of observations.
    kernel::Function: Kernel function (default: epanechnikov_kernel).

Returns:
    betas::Matrix: Matrix of posterior median estimates for regression coefficients at each time point.
"""
function QBLL_regression(y::AbstractVector, X::AbstractMatrix, n_iter::Int, H::Real; 
                        kernel::Function=epanechnikov_kernel)
    T = size(y,1)
    p = size(X,2)

    # Validate inputs
    if T < p + 1
        throw(DomainError("Not enough data points to estimate the model"))
    end
    if size(X, 1) != T
        throw(DimensionMismatch("Length of y must match number of rows in X"))
    end
    if n_iter <= 0
        throw(DomainError("Number of simulations n_iter must be positive"))
    end
    if H <= 0.0 || H > T
        throw(DomainError("Bandwidth must be in (0, T]"))
    end

    X = [ones(T) X]
    
    # Use full sample OLS estimates as priors...
    β0 = X \ y
    e = y - X * β0

    # Diagonal elements of this are the scale parameter of Gamma distribution.
    s2= e' * e / (T - p) 

    g0 = 1 / s2
    # Scale down prior precision to avoid overwhelming the local likelihood
    k0 = (1 / T) * (1 / s2) * inv(X' * X)
    a0 = max(p + 2, p + 2 * 8 - T)
    g0 = (a0 - p - 1) * g0

    weights = kernel_QBLL(T, H, kernel)

    # Storage for betas and lambdas
    betas = zeros(T, 1 + p);
    
    for ii in 1:T
        B1 = zeros(p+1,n_iter)
        
        w=weights[ii,:]
        bayesprec = k0 + X' * Diagonal(w) * X
        bayessv = inv(bayesprec)
        BB = bayessv * ((X' * Diagonal(w)) * y + k0 * β0)
        bayesalpha = a0 + sum(w) / 2

        g1 = β0' * k0 * β0
        g2 = y' * Diagonal(w) * y 
        g3 = BB' * bayesprec * BB
        bayesgamma = g0 + 0.5 * (g1 + g2 - g3)

        for ll in 1:n_iter
            B1[:,ll] = BB + cholesky(Symmetric(bayessv) .* 1.0).U' * randn(p + 1) * sqrt(rand(Gamma(bayesalpha,1 / bayesgamma)))
        end

        for i in 1:p+1
            betas[ii,i] = quantile(B1[i,:], 0.5)
        end

    end
    
    return betas
end

function QBLL_regression(y::AbstractVector, X::AbstractVector, n_iter::Int, H::Real; 
                        kernel::Function=epanechnikov_kernel)
    return QBLL_regression(y, reshape(X, :, 1), n_iter, H; kernel=kernel)
end

"""
    QBLL_ar(y::AbstractVector, p::Int, n_iter::Int, H::Real; kernel::Function=epanechnikov_kernel)

Performs Quasi-Bayesian Local Likelihood (QBLL) regression for autoregressive models.

Arguments:
    y::AbstractVector: Time series variable vector.
    p::Int: Autoregressive order (number of lags).
    n_iter::Int: Number of posterior simulations per time point.
    H::Real: Bandwidth parameter for kernel weighting from interval [1, T-p] where T is the number of observations.
    kernel::Function: Kernel function (default: epanechnikov_kernel).

Returns:
    betas::Matrix: Matrix of posterior median estimates for autoregressive coefficients at each time point.
"""
function QBLL_ar(y::AbstractVector, p::Int, n_iter::Int, H::Real; 
                    kernel::Function=epanechnikov_kernel)
    if p < 1
        throw(DomainError("AR order p must be at least 1"))
    end

    y_target, X = _create_ar_variables(y, p)
    betas = QBLL_regression(y_target, X, n_iter, H; kernel=kernel)

    return betas
end