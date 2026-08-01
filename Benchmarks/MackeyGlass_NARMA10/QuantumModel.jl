import QuantumOptics as QO
using LinearAlgebra
import Plots as plt
using Statistics
using QuantumToolbox
using Random
using Combinatorics
using StatsPlots

"""
Annihilation operators in fock basis: Nc levels
Two level atoms in their eigenbasis: Natoms
One atom by default
Return: undriven hamiltonian and the operaters creating it
"""
function system(;Nc::Int=10, Natom::Int=1, ω_c::Float64=0.0, ω_i::Vector{Float64}=[0.0], g_i=[sqrt(.05)])
    @assert length(ω_i) == Natom && length(g_i) == Natom
    # Define Hilbert Space
    b_cav = QO.FockBasis(Nc) # base 0, 1,2,3,..,Nc photons. Light's Hilbert space
    b_at = QO.SpinBasis(1//2) # 1/2 spin basis. Atom's Hilber Space
    b_total = QO.tensor(b_cav, fill(b_at, Natom)...) # tensorbasis: product state, fill: create multiple, ... a vector->multiple arguments
    cavity_position = 1

    #cavity operators
    c_local = QO.destroy(b_cav) # acting on only local Hilbert Space
    c = QO.embed(b_total, cavity_position, c_local) #lifting local operator into the total Hilbert space, subsystem cavity_position

    # Atom operators
    sigmas = Vector{QO.Operator}(undef, Natom) #operators type array, notdefined operator, for Natom
    for i in 1:Natom
        sigmas[i] = QO.embed(b_total, i+1, QO.sigmam(b_at)) #lifting to total Hilbert space and choose signma minus
    end

    # Undriven hamiltonian H_0 = ω_c ĉ†ĉ + ∑ω_isigma†sigma + ∑g_i(c†sigma+csigma†)
    H_0 = ω_c*QO.dagger(c)*c
    for i in 1:Natom
        H_0 += ω_i[i] * QO.dagger(sigmas[i]) * sigmas[i]
        H_0 += g_i[i] * (QO.dagger(c)*sigmas[i]+c*QO.dagger(sigmas[i]))
    end

    # intitialize the state
    ψ_c = QO.fockstate(b_cav, 0) # ground state cavity
    ψ_a = QO.spinup(b_at) # create excited atom 
    ψ = QO.tensor(ψ_c, fill(ψ_a, Natom)...)
    ρ0 = QO.dm(ψ)

    return c, sigmas, H_0, ρ0
end

# Driving hamiltonian H_drive = i ϵ f(t) (c - c†)
function H_drive(sigmas::Vector{QO.Operator}, ϵ)
    return im * ϵ  * (sigmas[1] - QO.dagger(sigmas[1]))
end

function lindblad_ops(c::QO.Operator, sigmas::Vector{QO.Operator}; κ)
    Ls = QO.Operator[] # Vector{Operator}()
    #push!(Ls, sqrt(κ) * c) #sqr
    for σ in sigmas
        push!(Ls, sqrt(κ_i) * σ)
    end
    return Ls
end

function lindblad_ops_complex(c::QO.Operator, sigmas::Vector{QO.Operator}; κ,h)
    Natom = length(sigmas)
    κ_i = κ
    Ls = QO.Operator[] # Vector{Operator}()
    #push!(Ls, c)
    for σ in sigmas
        push!(Ls, σ)
    end
    rates = fill(κ_i, 1)
    return Ls, rates
end


"""
Feature operators: cavity quadratures up to 10th-order polynomial
Returns:
    Degree 1:  Q, P
    Degree 2:  Q², QP, P²
    Degree 3:  Q³, Q²P, QP², P³
    ...
    Degree 10: Q¹⁰, Q⁹P, ..., P¹⁰

Total number of features:
    ∑_{d=1}^{10} (d+1) = 65
"""
function feature_ops(c::QO.Operator)
    Q = c + QO.dagger(c)
    P = im * (c - QO.dagger(c))
    ops = QO.Operator[]
    for total_degree in 1:10
        for qdeg in total_degree:-1:0
            pdeg = total_degree - qdeg
            op = one(Q)   # Identity operator
            for _ in 1:qdeg
                op *= Q
            end
            for _ in 1:pdeg
                op *= P
            end
            push!(ops, op)
        end
    end
    return ops
end


"""
This function takes an input and out put the nodes
"""
function state_evolution(f::AbstractVector,ϵ,κ,h1,option;Nc::Int = 10,Natom::Int = 1,ω_c::Float64 = 0.0,ω_i = [0.0],g_i = [sqrt(.05)],total_points = length(f),t_max = 5.0)
    # Build system
    c, sigmas, H_0, ρ0 = system(
        Nc = Nc,
        Natom = Natom,
        ω_c = ω_c,
        ω_i = ω_i,
        g_i = g_i
    )

    # Measurement operators
    ops = feature_ops(c)
    Nnodes = length(ops)

    # Lindblad operators
    Ls, rates = lindblad_ops_complex(c, sigmas, κ = κ, h = h1)
    Lsdagger = QO.dagger.(Ls)

    K = length(f)
    Δt = t_max / K
    tlist = collect(0:Δt:(t_max - 1e-9))

    function H_total(t, ρ)
        i = Int(trunc(t / Δt + 1))
        i = min(i, length(f))

        H = H_0 + f[i] * H_drive(sigmas, ϵ)

        return H, Ls, Lsdagger, rates
    end

    tout, ρt = QO.timeevolution.master_dynamic(tlist, ρ0, H_total)

    # Convert QuantumOptics density operators into matrices
    ρ_matrices = [Matrix(ρ.data) for ρ in ρt]

    if option == 1
        X = zeros(Float64, total_points, Nnodes)

        n_op = QO.dagger(c) * c
        NN = Float64[]

        for k in 1:total_points
            ρ = ρt[k]

            for j in 1:Nnodes
                X[k, j] = real(QO.expect(ops[j], ρ))
            end

            push!(NN, real(QO.expect(n_op, ρ)))
        end

        return ρ_matrices, X, NN

    else
        return ρ_matrices
    end
end
