"""
Target output: Ȳ : Lx1 column matrix, L is the amount of training points
Nodes matrix: X : Lx(Nnodes + 1) where 1 is a constant bias
Optimized weight matrix: W = X†Ȳ : nnodes + 1
where X† = (X^T X + δI)^-1 * X^T
δ=10e-10
y_vec = w_0 + ∑w_i*x_vec_i
"""
function fittingFunction(X::Matrix{Float64}, y::Vector{Float64}; δ=1e-10) 
    L, nnodes = size(X)
    Xb = hcat(ones(L), X) # adding a bias column/ horizontal cat
    A = Xb'Xb + δ*I(nnodes+1)
    W = A \ (Xb'*y)
    return W
end

function predict(X::Matrix{Float64}, W::Vector{Float64})
    return hcat(ones(size(X)[1]),X) * W
end

function NRMSE(ypred::Vector{Float64}, ytarget::Vector{Float64})
    denom = maximum(ytarget) - minimum(ytarget)
    return sqrt(mean((ypred .- ytarget).^2))/ denom
end

