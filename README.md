# TVModels 

[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A Julia package for nonparametric and semi-parametric modeling of time-varying relationships in time series. TVModels combines rolling and kernel regression methods, state space models, and Bayesian approaches to flexibly capture gradual structural changes, nonstationarity, and evolving dynamics. The package provides tools for estimation, bandwidth/kernel selection, forecasting, and simulation of synthetic series.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/B0B36JUL-FinalProjects-2025/Project_hukovboh")
```

## Methods

### 1. Rolling Window Regression
Fixed or expanding window OLS estimation for time-varying coefficients.

- **`rolling_regression(y, X, window_size; expanding=false)`** - General regression
- **`rolling_ar(y, p, window_size; expanding=false)`** - AR(p) model

### 2. Kernel Regression Methods
Nonparametric estimation using kernel weighting with various kernel functions.

**Available Kernels:**
- `epanechnikov_kernel`, `uniform_kernel`, `triangular_kernel`
- `gaussian_kernel`, `cosine_kernel`, `logistic_kernel`  
- `sigmoid_kernel`, `silverman_kernel`

**Estimation Functions:**
- **`kernel_local_level(y, bandwidth; kernel)`** - Local level smoothing
- **`kernel_local_linear(y, bandwidth; kernel)`** - Local linear trend
- **`kernel_regression(y, X, bandwidth; kernel)`** - General regression
- **`kernel_ar(y, p, bandwidth; kernel)`** - AR(p) model
- **`lowess_regression(y, X, bandwidth, n_iter; kernel)`** - LOWESS with robust weighting
- **`lowess_ar(y, p, bandwidth, n_iter; kernel)`** - LOWESS AR(p)

### 3. State Space Models
Kalman filtering and smoothing for time-varying parameter estimation. 

*Our implementation allows custom specification of state-transition equations.*

- **`state_local_level(y; sigma_eps, sigma_eta, μ0, P0, smooth)`** - Local level model
- **`state_regression(y, X; sigma_eps, Q, β0, P0, smooth)`** - TVP regression
- **`state_ar(y, p; sigma_eps, Q, β0, P0, smooth)`** - State space AR(p)

Returns: `β_filt, P_filt, β_smooth, P_smooth` (filtered and smoothed estimates)

### 4. Bayesian Methods
Quasi-Bayesian Local Likelihood (QBLL) for posterior inference.

- **`QBLL_regression(y, X, n_iter, H; kernel)`** - General regression
- **`QBLL_ar(y, p, n_iter, H; kernel)`** - AR(p) model

Returns posterior median estimates at each time point.

### 5. Parameter Optimization
Grid search with cross-validation for bandwidth/parameter selection.

- **`optimize_kernel_local(y; bandwidths, kernels, folds, metric)`**
- **`optimize_kernel_regression(y, X; bandwidths, kernels, folds, metric)`**
- **`optimize_kernel_ar(y, p; bandwidths, kernels, folds, metric)`**
- **`optimize_lowess_regression(y, X; bandwidths, n_iters, kernels, folds, metric)`**
- **`optimize_lowess_ar(y, p; bandwidths, n_iters, kernels, folds, metric)`**
- **`optimize_QBLL_regression(y, X; Hs, n_iters, kernels, folds, metric)`**
- **`optimize_QBLL_ar(y, p; Hs, n_iters, kernels, folds, metric)`**

### 6. Forecasting
Multi-step ahead forecasting for time-varying models.

- **`forecast_regression(X, β)`** - One-step ahead forecast
- **`forecast_ar(X, β, h)`** - Multi-step AR forecast
- **`forecast_state_ar(X, β_filt, P_filt, h; Q, state_transition)`** - State space AR forecast

### 7. Simulation
Generate synthetic time-varying series for testing.

- **`simulate_tv_ar(T, βs; σ, seed)`** - TV-AR with matrix of coefficients
- **`simulate_tv_ar(T, β_funcs; σ, seed)`** - TV-AR with coefficient functions
- **`simulate_local_level(T, sigma_eps, sigma_eta; μ0, seed)`** - Local level model

## Example: Comparing Methods

```julia
using TVModels

# Simulate time-varying AR(1) process
y = simulate_tv_ar(200, [t -> 0.5 + 0.3*sin(t), t -> 0.6*cos(t)]; σ=0.1)

# Estimate with different methods
betas_rolling = rolling_regression(y, 1, 30; expanding=false)
betas_kernel = kernel_ar(y, 1, 0.05; kernel=epanechnikov_kernel)
betas_lowess = lowess_ar(y, 1, 0.03, 3; kernel=silverman_kernel)
betas_state, _, betas_state_smooth, _ = state_ar(y, 1; sigma_eps=0.5, Q=0.01, β0=[0.0, 0.0], P0=1.0)
betas_qbll = QBLL_ar(y, 1, 1000, 30)
```
