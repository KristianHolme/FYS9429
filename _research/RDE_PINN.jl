using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, LineSearches,
      OptimizationOptimisers
using ModelingToolkit: Interval, infimum, supremum
using RDE, RDE_Env
using ProgressMeter

@parameters t, x
@variables u(..), λ(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2
##
RDEParams = RDEParam{Float64}(tmax = 1.0)
q_0 = RDEParams.q_0
ν_1 = RDEParams.ν_1
ν_2 = RDEParams.ν_2
L = RDEParams.L
u_c = RDEParams.u_c
α = RDEParams.α
u_0 = RDEParams.u_0
n = RDEParams.n
s = RDEParams.s
u_p = RDEParams.u_p
k = RDEParams.k_param
ϵ = RDEParams.ϵ
tmax = RDEParams.tmax
##
ω(u) = RDE.ω(u, u_c, α)
ξ(u) = RDE.ξ(u, u_0, n)
β(u) = RDE.β(u, s, u_p, k)


eqs = [
    Dt(u(t, x)) + u(t, x) * Dx(u(t, x)) ~ (1 - λ(t, x)) * ω(u(t, x)) * q_0 + ν_1 * Dxx(u(t, x)) + ϵ * ξ(u(t, x)),
    Dt(λ(t, x)) ~ (1 - λ(t, x)) * ω(u(t, x)) - β(u(t, x)) * λ(t, x) + ν_2 * Dxx(λ(t, x))
]

bcs = [
    u(0, x) ~ RDE.default_u(x),
    λ(0, x) ~ RDE.default_λ(x),
    u(t, 0) ~ u(t, L),
    λ(t, 0) ~ λ(t, L)
]

domains = [t ∈ Interval(0.0, tmax), x ∈ Interval(0.0, L)]

# Neural network
input_ = length(domains)
n = 512
chain = [Chain(Dense(input_, n, σ), Dense(n, n, σ), Dense(n, 1)) for _ in 1:2]

# strategy = StochasticTraining(128)
strategy = QuasiRandomTraining(128)
discretization = PhysicsInformedNN(chain, strategy, adaptive_loss = GradientScaleAdaptiveLoss(1024))

@named pdesystem = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), λ(t, x)])
prob = discretize(pdesystem, discretization)
sym_prob = symbolic_discretize(pdesystem, discretization)

pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions

progressbar = ProgressUnknown("Training...")
callback = function (p, l)
    next!(progressbar, showvalues = [(:loss, l), (:pde_losses, map(l_ -> l_(p.u), pde_inner_loss_functions)), (:bcs_losses, map(l_ -> l_(p.u), bcs_inner_loss_functions))])
    return false
end
##
opt = OptimizationOptimisers.Adam(3e-4)
res = solve(prob, opt; maxiters = 1000, callback)
## Train more
prob = remake(prob, u0 = res.u)
res = solve(prob, opt; maxiters = 10_000, callback)

## post-processing and plotting
phi = discretization.phi
ts, xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]

minimizers_ = [res.u.depvar[sym_prob.depvars[i]] for i in 1:2]

predict_u(t, x) = phi[1]([t, x], minimizers_[1])[1]
predict_λ(t, x) = phi[2]([t, x], minimizers_[2])[1]

##
using CairoMakie
us = [predict_u.(t, xs) for t in ts]
λs = [predict_λ.(t, xs) for t in ts]
env = RDEEnv(RDEParams, dt = 0.01)
sim_data = run_policy(ConstantRDEPolicy(env), env)
ts_sim = sim_data.state_ts
xs_sim = env.prob.x
us_sim, λs_sim = RDE.split_sol(sim_data.states)
##
u_min, u_max = minimum(minimum.(us)), maximum(maximum.(us))
u_min  = min(u_min, minimum(minimum.(us_sim)))
u_max = max(u_max, maximum(maximum.(us_sim)))
λ_min, λ_max = minimum(minimum.(λs)), maximum(maximum.(λs))
λ_min  = min(λ_min, minimum(minimum.(λs_sim)))
λ_max = max(λ_max, maximum(maximum.(λs_sim)))
##
fig = Figure(size = (1000, 1000))
ax_u = Axis(fig[1, 1], title = "NN u(t, x)", xlabel = "t", ylabel = "x")
heatmap!(ax_u, ts, xs, stack(us)', colorrange = (u_min, u_max))
ax_λ = Axis(fig[1, 2], title = "NN λ(t, x)", xlabel = "t", ylabel = "x")
heatmap!(ax_λ, ts, xs, stack(λs)', colorrange = (λ_min, λ_max))
ax_u_sim = Axis(fig[2, 1], title = "Sim u(t, x)", xlabel = "t", ylabel = "x")
heatmap!(ax_u_sim, ts_sim, xs_sim, stack(us_sim)', colorrange = (u_min, u_max))
ax_λ_sim = Axis(fig[2, 2], title = "Sim λ(t, x)", xlabel = "t", ylabel = "x")
heatmap!(ax_λ_sim, ts_sim, xs_sim, stack(λs_sim)', colorrange = (λ_min, λ_max))
fig