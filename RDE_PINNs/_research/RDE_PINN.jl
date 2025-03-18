using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, LineSearches,
OptimizationOptimisers
using ComponentArrays
using ModelingToolkit: Interval, infimum, supremum
using RDE, RDE_Env
using CairoMakie
using ProgressMeter
using Random
using LuxCUDA
using CUDA
##
CUDA.allowscalar(true)
##
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
    u(0, x) ~ RDE.default_u(x)*1.5,
    λ(0, x) ~ 0.5,
    u(t, 0) ~ u(t, L),
    λ(t, 0) ~ λ(t, L)
]

domains = [t ∈ Interval(0.0, tmax), x ∈ Interval(0.0, L)]

## Neural network
rng = Random.default_rng()
gdev = gpu_device(1)
cdev = cpu_device()
input_ = length(domains)
n = 64
layers = 8
chain = [Chain(Dense(input_, n, tanh), [Dense(n, n, tanh) for i in layers]..., Dense(n, 1)) for _ in 1:2]
## For GPU
ps = Lux.setup(rng, chain)[1] .|> ComponentArray .|> gdev
ps = [p .|> Float64 for p in ps]
## For CPU
ps = Lux.setup(rng, chain)[1]
##

# strategy = StochasticTraining(128)
strategy = QuasiRandomTraining(2^14)
# discretization = PhysicsInformedNN(chain, strategy, adaptive_loss = GradientScaleAdaptiveLoss(64), init_params = ps)
discretization = PhysicsInformedNN(chain, strategy, adaptive_loss = MiniMaxAdaptiveLoss(64), init_params = ps)


@named pdesystem = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), λ(t, x)])
prob = discretize(pdesystem, discretization)
sym_prob = symbolic_discretize(pdesystem, discretization)

pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions


losses = Float64[]
n_pde_losses = length(pde_inner_loss_functions)
n_bcs_losses = length(bcs_inner_loss_functions)
pde_losses = Float64[]
bcs_losses = Float64[]
progressbar = ProgressUnknown("Training...", showspeed = true)
callback = function (p, l)
    if (progressbar.core.counter +1) % 1 == 0 || progressbar.core.counter == 0
        push!(losses, l)
        push!(pde_losses, map(l_ -> l_(p.u), pde_inner_loss_functions)...)
        push!(bcs_losses, map(l_ -> l_(p.u), bcs_inner_loss_functions)...)
    end
    next!(progressbar, showvalues = [(:loss, losses[end]),
     (:pde_losses, pde_losses[end-n_pde_losses+1:end]),
     (:bcs_losses, bcs_losses[end-n_bcs_losses+1:end])])
    # next!(progressbar, showvalues = [(:loss, l)])
    return false
end
##
opt = OptimizationOptimisers.Adam(3e-4)
# opt = BFGS(linesearch = BackTracking())
res = solve(prob, opt; maxiters = 50, callback);
@info "Loss = $(res.objective)"
## Train more
opt = OptimizationOptimisers.Adam(0.01)
# opt = OptimizationOptimJL.LBFGS()
prob = remake(prob, u0 = res.u)
res = solve(prob, opt; maxiters = 200, callback);
@info "Loss = $(res.objective)"
## Plot losses
pde_u_losses = stack(pde_losses)[1,:]
pde_λ_losses = stack(pde_losses)[2,:]
ic_u_losses = stack(bcs_losses)[1,:]
ic_λ_losses = stack(bcs_losses)[2,:]
fig = Figure()
ax_losses = Makie.Axis(fig[1, 1], title = "Losses", xlabel = "Iteration", ylabel = "Loss", xscale = log10, yscale = log10)
lines!(ax_losses, losses)
ax_pde_losses = Makie.Axis(fig[1, 2], title = "PDE Losses", xlabel = "Iteration", ylabel = "Loss", xscale = log10, yscale = log10)
lines!(ax_pde_losses, pde_u_losses, label = "u")
lines!(ax_pde_losses, pde_λ_losses, label = "λ")
ax_bcs_losses = Makie.Axis(fig[2, 1], title = "BCs Losses", xlabel = "Iteration", ylabel = "Loss", xscale = log10, yscale = log10)
u_line = lines!(ax_bcs_losses, ic_u_losses, label = "u")
λ_line = lines!(ax_bcs_losses, ic_λ_losses, label = "λ")
Legend(fig[2,2], [u_line, λ_line], ["u", "λ"], tellwidth = false, tellheight = false)
fig

## post-processing and plotting
phi = discretization.phi
ts, xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]

minimizers_ = [res.u.depvar[sym_prob.depvars[i]] for i in 1:2]

predict_u(t, x) = phi[1]([t, x], minimizers_[1])[1]
predict_λ(t, x) = phi[2]([t, x], minimizers_[2])[1]

#
us = [predict_u.(t, xs) for t in ts]
λs = [predict_λ.(t, xs) for t in ts]
env = RDEEnv(RDEParams, dt = 0.01)
sim_data = run_policy(ConstantRDEPolicy(env), env)
ts_sim = sim_data.state_ts
xs_sim = env.prob.x
us_sim, λs_sim = RDE.split_sol(sim_data.states)
#
u_min, u_max = minimum(minimum.(us)), maximum(maximum.(us))
u_min  = min(u_min, minimum(minimum.(us_sim)))
u_max = max(u_max, maximum(maximum.(us_sim)))
λ_min, λ_max = minimum(minimum.(λs)), maximum(maximum.(λs))
λ_min  = min(λ_min, minimum(minimum.(λs_sim)))
λ_max = max(λ_max, maximum(maximum.(λs_sim)))

fig = Figure(size = (1000, 1000))
ax_u = Axis(fig[1, 1], title = "PINN u(t, x)", xlabel = "t", ylabel = "x")
hm_u = heatmap!(ax_u, ts, xs, stack(us)', colorrange = (u_min, u_max))
ax_λ = Axis(fig[1, 2], title = "PINN λ(t, x)", xlabel = "t", ylabel = "x")
hm_λ = heatmap!(ax_λ, ts, xs, stack(λs)', colorrange = (0, 1.1))
ax_u_sim = Axis(fig[2, 1], title = "Sim u(t, x)", xlabel = "t", ylabel = "x")
heatmap!(ax_u_sim, ts_sim, xs_sim, stack(us_sim)', colorrange = (u_min, u_max))
ax_λ_sim = Axis(fig[2, 2], title = "Sim λ(t, x)", xlabel = "t", ylabel = "x")
heatmap!(ax_λ_sim, ts_sim, xs_sim, stack(λs_sim)', colorrange = (0, 1.1))
Colorbar(fig[3, 1], hm_u, vertical = false)
Colorbar(fig[3, 2], hm_λ, vertical = false)
ax_u_t = Axis(fig[4, 1], title = "PINN u(t, x)", xlabel = "x", ylabel = "u")
N = length(ts)
l_start = lines!(ax_u_t, xs, us[1], label="t=0")
l_mid = lines!(ax_u_t, xs, us[Int(round(N/2))], label="t=$(ts[Int(round(N/2))])")
l_end = lines!(ax_u_t, xs, us[end], label="t=$(ts[end])")
ax_λ_t = Axis(fig[4, 2], title = "PINN λ(t, x)", xlabel = "x", ylabel = "λ")
lines!(ax_λ_t, xs, λs[1])
lines!(ax_λ_t, xs, λs[Int(round(N/2))])
lines!(ax_λ_t, xs, λs[end])
ylims!(ax_λ_t, 0, 1.1)
N_sim = length(ts_sim)
ax_u_sim_t = Axis(fig[5, 1], title = "Sim u(t, x)", xlabel = "x", ylabel = "u")
linkaxes!(ax_u_sim_t, ax_u_t)
lines!(ax_u_sim_t, xs_sim, us_sim[1])
lines!(ax_u_sim_t, xs_sim, us_sim[Int(round(N_sim/2))])
lines!(ax_u_sim_t, xs_sim, us_sim[end])
ax_λ_sim_t = Axis(fig[5, 2], title = "Sim λ(t, x)", xlabel = "x", ylabel = "λ")
linkaxes!(ax_λ_sim_t, ax_λ_t)
lines!(ax_λ_sim_t, xs_sim, λs_sim[1])
lines!(ax_λ_sim_t, xs_sim, λs_sim[Int(round(N_sim/2))])
lines!(ax_λ_sim_t, xs_sim, λs_sim[end])
ylims!(ax_λ_sim_t, 0, 1.1)
Legend(fig[end+1, :], [l_start, l_mid, l_end], ["t=0", "t=$(ts[Int(round(N/2))])", "t=$(ts[end])"], orientation=:horizontal, tellwidth=false, tellheight=true)

fig
