using ForwardDiff
using LinearAlgebra
import QuantumOptics as QO
using Plots

include("QuantumModel.jl")
include("Tasks.jl")
include("FittingModel.jl")

struct ReservoirParams
    ϵ::Float64
    g::Float64
    κ::Float64
end

# ----------------------------
# System Parameters
# ----------------------------
tmax = 30000
Δt= 1
γ = .1
ϕ = π/3
ϕ_prime = π - ϕ
Nc = 10
ω_c = 0.0
Natom = 1
ω_i = [0.0+γ/2*sin(ϕ_prime)]
p0 = ReservoirParams(
    .16/2,   # ϵ
    γ * ( 1 + cos(ϕ_prime))/4,   # g
    γ * ( 1 + cos(ϕ_prime))  # κ
)
ϵ = p0.ϵ
g_i = [p0.g]
κ = p0.κ


# ============================================================
# NARMA Task Setup
# ============================================================

Ntotal = 1000
Nfading = 50
Ntraining = 750
# Generate Mackey-Glass signal
using CSV
using DataFrames

df = CSV.read("NARMA10.csv", DataFrame)
u = Vector{Float64}(df.input)
f = Vector{Float64}(df.target)
#f = mackey_glass(Ntotal; delay=τ)
# Input to reservoir
# Total points actually used by reservoir
total_points = length(u)
# Target: predict τ steps ahead
y_target = f[:]


# Because y_target is shorter than f by τ,
# we should also align reservoir features later by using only first K rows.
ρ_base, X, NN = state_evolution(
    u,
    ϵ,
    κ,
    0.0,
    1;
    Nc = Nc,
    Natom = Natom,
    ω_c = ω_c,
    ω_i = ω_i,
    g_i = g_i,
    total_points = total_points,
    t_max = tmax
)

X_aligned = X[:, 1:50]
Ntesting = Ntotal - Nfading - Ntraining
train_start = Nfading + 1
train_end   = Nfading + Ntraining

test_start = Nfading + Ntraining + 1
test_end   = Nfading + Ntraining + Ntesting

X_train = X_aligned[train_start:train_end, :]
X_test  = X_aligned[test_start:test_end, :]

y_train = y_target[train_start:train_end]
y_test  = y_target[test_start:test_end]
W = fittingFunction(X_train, y_train)

y_prediction = predict(X_test, W)

nrmse_mg = NRMSE(y_prediction, y_test)
Plots.plot(y_test, label="target", lw=2)
Plots.plot!(y_prediction, label="prediction", lw=2)


using CSV
using DataFrames

colnames = ["X_$i" for i in 1:size(X, 2)]
df = DataFrame(X, Symbol.(colnames))
CSV.write("NM10_epsilon0.16.csv", df)