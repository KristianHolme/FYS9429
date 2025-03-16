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
##
RDEParams = RDEParam(tmax = 400.0f0)
env = RDEEnv(RDEParams,
    dt = 1.0f0,
    observation_strategy=SectionedStateObservation(minisections=32),
    reset_strategy=ShiftReset(RandomShockOrCombination()))
N = RDEParams.N
policies = [ConstantRDEPolicy(env), 
 ScaledPolicy(SinusoidalRDEPolicy(env, w_1=0f0, w_2=0.5f0), 0.5f0),
 ScaledPolicy(RandomRDEPolicy(env), 0.5f0),
 ScaledPolicy(RandomRDEPolicy(env), 0.2f0),
 load_best_policy("transition_rl_8", project_path=joinpath(homedir(), "Code", "DRL_RDE"))[1]
]
data = []
n_runs = 10
prog = Progress(n_runs*length(policies), "Collecting data...")
for policy in policies, i in 1:n_runs
    sim_data = run_policy(policy, env)
    n_data = length(sim_data.states)

    raw_data = zeros(Float32, N, 3, n_data)
    x_data = @view raw_data[:,:,1:end-1]
    y_data = @view raw_data[:,1:2,2:end]

    for i in eachindex(sim_data.observations)
        obs = sim_data.states[i]
        raw_data[:, 1, i] = obs[1:N]
        raw_data[:, 2, i] = obs[N+1:2N]
        raw_data[:, 3, i] .= sim_data.u_ps[i]
    end
    push!(data, (x_data, y_data))
    next!(prog, showvalues=[("Collected data sets for $(typeof(policy))", i)])
end


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
const gdev = gpu_device()
# const xdev = reactant_device()

fno_gpu = FourierNeuralOperator(gelu; chs=(3, 64, 64, 128, 2), modes=(16,), permuted=Val(true))

ps_gpu, st_gpu = Lux.setup(rng, fno_gpu) |> gdev;
@time losses_gpu = train!(fno_gpu, ps_gpu, st_gpu, data; epochs=100, dev=gdev)
##more training
@time losses_gpu = train!(fno_gpu, ps_gpu, st_gpu, data;
     losses=losses_gpu, epochs=100, dev=gdev, lr=3f-4)

##
fig = Figure()
ax = Makie.Axis(fig[1, 1], xlabel="Epoch", ylabel="Loss", xscale=log10, yscale=log10)

# Calculate smoothed line and confidence bands
window_size = 50
smoothed = [mean(losses_gpu[max(1,i-window_size):i]) for i in 1:length(losses_gpu)]
upper = [quantile(losses_gpu[max(1,i-window_size):i], 0.95) for i in 1:length(losses_gpu)]
lower = [quantile(losses_gpu[max(1,i-window_size):i], 0.05) for i in 1:length(losses_gpu)]

# Plot bands and smoothed line
band!(ax, 1:length(losses_gpu), lower, upper, color=(:blue, 0.2))
lines!(ax, smoothed, color=:blue, linewidth=2)
fig
## Test the model
RDEParams = RDEParam(tmax = 800.0)
env = RDEEnv(RDEParams,
    dt = 1.0,
    observation_strategy=SectionedStateObservation(minisections=32))
sim_test_data = run_policy(ScaledPolicy(RandomRDEPolicy(env), 0.2f0), env)
plot_shifted_history(sim_test_data, env.prob.x)
test_states = sim_test_data.states[400:end]

test_data = zeros(Float32, N, 3, length(test_states))
for i in eachindex(test_states)
    obs = test_states[i]
    test_data[:, 1, i] = obs[1:N]
    test_data[:, 2, i] = obs[N+1:2N]
    test_data[:, 3, i] .= sim_test_data.u_ps[i]
end

output_data, st = Lux.apply(fno_gpu, test_data[:,:,1:end-1] |> gdev, ps_gpu, st_gpu) |> cdev
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

