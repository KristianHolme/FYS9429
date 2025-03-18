using DrWatson
@quickactivate "RDE_PINNs"
using DRL_RDE_utils
using ProgressMeter
using NeuralOperators
using Lux
using LuxCUDA
using Optimisers
using Zygote
using LinearAlgebra
using Random
using OptimizationOptimisers
using CairoMakie
using JLD2
using Statistics
using Base.Threads
##
policies, envs = make_data_policies_and_envs()
##
policy = policies[end]
policy.py_policy.action_space
policy.py_policy.predict(rand(Float32, 66), deterministic=true)
policy.py_policy
##
n_runs = 16
run_datas = Vector{Any}(undef, length(policies)*n_runs)
data = Vector{Tuple{Array{Float32, 3}, Array{Float32, 3}}}(undef, length(policies)*n_runs)
prog = Progress(n_runs*length(policies), "Collecting data...")
data_collect_stats = zeros(Int, length(policies), n_runs)
for i in 1:(length(policies)*n_runs)
    policy_idx = div(i-1, n_runs) + 1
    run_idx = mod(i-1, n_runs) + 1
    policy = policies[policy_idx]
    env = envs[policy_idx]
    
    sim_data = run_policy(policy, env)
    run_datas[i] = sim_data
    
    n_data = length(sim_data.states)
    raw_data = zeros(Float32, N, 3, n_data)
    x_data = @view raw_data[:,:,1:end-1]
    y_data = @view raw_data[:,1:2,2:end]
    
    for j in eachindex(sim_data.observations)
        obs = sim_data.states[j]
        raw_data[:, 1, j] = obs[1:N]
        raw_data[:, 2, j] = obs[N+1:2N]
        raw_data[:, 3, j] .= sim_data.u_ps[j]
    end
    data[i] = (x_data, y_data)
    data_collect_stats[policy_idx, run_idx] += 1
    next!(prog, showvalues=[("$(typeof(policies[ip]))", sum(data_collect_stats[ip,:])) for ip in eachindex(policies)])
end
## inspecting data
for (i, rdata) in enumerate(run_datas)
    policy_idx = div(i-1, n_runs) + 1
    run_idx = mod(i-1, n_runs) + 1
    fig = plot_shifted_history(rdata, envs[policy_idx].prob.x, title="Run $run_idx, Policy $policy_idx")
    display(fig)
end
## Saving data
jldsave(datadir("data.jld2"); data)
## Loading data
data = load(datadir("data.jld2"))["data"]
##
function train!(model, ps, st, data; lr = 3f-4, epochs=10, losses=[], dev=cpu_device(), AD=AutoZygote())
    tstate = Training.TrainState(model, ps, st, OptimizationOptimisers.Adam(lr))
    p = Progress(epochs*length(data))
    for _ in 1:epochs, (x, y) in dev(data)
        _, loss, _, tstate = Training.single_train_step!(AD, MSELoss(), (x, y),
        tstate)
        push!(losses, loss)
        next!(p, showvalues=[("Loss", loss)])
    end
    return losses
end


# Save the model parameters and state
# @save "fno_model.jld2" fno ps st

## GPU training
rng = Random.default_rng()
const cdev = cpu_device()
const gdev = gpu_device(2)
# const xdev = reactant_device()

fno = FourierNeuralOperator(gelu; chs=(3, 128, 128, 128, 128, 2), modes=(32,), permuted=Val(true))

ps, st = Lux.setup(rng, fno) |> gdev;

@time losses = train!(fno, ps, st, data; epochs=100, dev=gdev, lr=0.001f0)
##more training
@time losses = train!(fno, ps, st, data;
     losses=losses, epochs=100, dev=gdev, lr=3f-5)

##
fig = Figure()
ax = Makie.Axis(fig[1, 1], xlabel="Epoch", ylabel="Loss", xscale=log10, yscale=log10)

# Calculate smoothed line and confidence bands
window_size = 50
smoothed = [mean(losses[max(1,i-window_size):i]) for i in 1:length(losses)]
upper = [quantile(losses[max(1,i-window_size):i], 0.95) for i in 1:length(losses)]
lower = [quantile(losses[max(1,i-window_size):i], 0.05) for i in 1:length(losses)]

# Plot bands and smoothed line
band!(ax, 1:length(losses), lower, upper, color=(:blue, 0.2))
lines!(ax, smoothed, color=:blue, linewidth=2)
fig
## Test the model
RDEParams = RDEParam(tmax = 800.0)
env = RDEEnv(RDEParams,
    dt = 1.0,
    observation_strategy=SectionedStateObservation(minisections=32))
sim_test_data = run_policy(StepwiseRDEPolicy(env, [20.0f0, 100.0f0, 200.0f0, 300.0f0, 400.0f0, 500.0f0, 600.0f0, 700.0f0], 
[0.64f0, 0.96f0, 0.45f0, 0.84f0, 0.5f0, 0.75f0, 0.4f0, 0.62f0]), env)
plot_shifted_history(sim_test_data, env.prob.x)
test_states = sim_test_data.states[400:end]

test_data = zeros(Float32, N, 3, length(test_states))
for i in eachindex(test_states)
    obs = test_states[i]
    test_data[:, 1, i] = obs[1:N]
    test_data[:, 2, i] = obs[N+1:2N]
    test_data[:, 3, i] .= sim_test_data.u_ps[i]
end

@time output_data, st = Lux.apply(fno, test_data[:,:,1:end-1] |> gdev, ps, st) |> cdev
##
fig = Figure()
ax_u = Makie.Axis(fig[1, 1], xlabel="Time", ylabel="u")
ax_λ = Makie.Axis(fig[2, 1], xlabel="Time", ylabel="λ")
ls = []
n_test = length(test_states)
colors = Makie.wong_colors()[1:3]
for (i, ind) in enumerate([1, 2, n_test÷2])
    u_true = lines!(ax_u, test_data[:, 1, ind+1], color=colors[i], linestyle=:dash)
    u_pred = lines!(ax_u, output_data[:, 1, ind], color=colors[i])
    λ_true = lines!(ax_λ, test_data[:, 2, ind+1], color=colors[i], linestyle=:dash)
    λ_pred = lines!(ax_λ, output_data[:, 2, ind], color=colors[i])
    if ind == 1
        push!(ls, u_true)
        push!(ls, u_pred)
        push!(ls, λ_true)
        push!(ls, λ_pred)
    end
end
Legend(fig[:,2], ls[1:2], ["True", "Predicted"], vertical=true, tellheight=false)
fig
##
save(plotsdir("fno_test_modes16.png"), fig)
##
# Calculate test loss
test_loss = mean(abs2, output_data .- test_data[:,:,2:end])
@info "Test Loss: $test_loss"

