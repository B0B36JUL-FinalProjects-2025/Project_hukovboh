
# Create target vector and design matrix for AR model
function _create_ar_variables(y::AbstractVector, p::Int)
    T = length(y)
    X = ones(T - p, p)
    for j in 1:p
        X[:, j] = y[(p - j + 1):(end - j)]
    end
    y_target = y[(p + 1):end]
    return y_target, X
end