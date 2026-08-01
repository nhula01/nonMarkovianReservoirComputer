#------------
### ESN ####
#------------
function next_x(x::Vector{Float64}, A::Matrix{Float64}, B::Vector{Float64}, u::Float64)
    z = (A * x) .+ B .* u
    return tanh.(z)
end
function ESN_dynamics(inputs::Vector{Float64}, A::Matrix{Float64}, B::Vector{Float64})
    inputs = inputs#s.*2.0 .- 1.0
    T = length(inputs) # number of inputs
    N = size(A, 1) # number of reservoir nodes

    x = 0.2 .* rand(N) .- 0.1 # random initial state of reservoir

    X = zeros(Float64, T, N)
    for t in 1:T
        x = next_x(x, A, B, inputs[t])
        X[t, :] .= x
    end
    return X
end
function make_esn_weights(N::Int;
                          input_scale::Float64=1.0,
                          spectral_radius::Float64=1.0,
                          connectivity::Float64=1.0)

    A = 2.0 .* rand(N, N) .- 1.0 # random recurrent weights [-1,1]

    if connectivity < 1.0
        mask = rand( N, N) .< connectivity
        A .*= mask
    end

    vals = eigvals(A)
    ρ = maximum(abs.(vals))
    if ρ > 0
        A .*= spectral_radius / ρ 
    end

    B = input_scale .* (2.0 .* rand(N) .- 1.0)
    println("A norm", norm(A))
    println("B norm", norm(B))
    println("A/B ratio", norm(A)/norm(B))
    return A, B
end 


#------------
### Mackey ####
#------------

"""
This function returns Npoints mackey values 
"""
function mackey_glass(Npoints::Int=1000; delay::Int=10,τ::Int=17, x_0::Float64=1.0, β::Float64=.2, γ::Float64=.1, n::Int=10, dt::Float64=1.0)
    x_set = [x_0]
    for i in 1:Npoints+1000 # this 1000 to get the function going/ not what we collect
        if i<=τ
            xpast = 0
        else
            xpast = x_set[i-τ]
        end
        x_0 += dt * (β * xpast / (1+xpast^n) - γ*x_0)
        push!(x_set, x_0)
    end
    return x_set[1002-delay:end]
end
