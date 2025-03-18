using DrWatson
@quickactivate "RDE_PINNs"
using DRL_RDE_utils
using RDEML
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
reset_strategies = make_data_reset_strategies()
##
n_runs_per_reset_strategy = 4
run_data, data = collect_data(policies, envs, reset_strategies; n_runs_per_reset_strategy)
save_data(run_data, data; filename="RDE_sim_data.jld2")
## Viz
visualize_data(run_data, policies, envs, reset_strategies, n_runs_per_reset_strategy; save_plots=true)
## Loading data
all_data = load(datadir("RDE_sim_data.jld2"))
run_data = all_data["run_data"]
data = all_data["data"]
## GPU training
rng = Random.default_rng()
const cdev = cpu_device()
const gdev = gpu_device(2)
# const xdev = reactant_device()

fno_config = FNOConfig()
fno = FNO(fno_config)

ps, st = Lux.setup(rng, fno) |> gdev;

@time losses = train!(fno, ps, st, data; epochs=100, dev=gdev, lr=0.001f0)
##more training
@time losses = train!(fno, ps, st, data;
     losses=losses, epochs=40, dev=gdev, lr=3f-4)

##
plot_losses(fno_config, losses)
## Test the model
##
RDEParams = RDEParam(tmax = 800.0)
env = RDEEnv(RDEParams,
    dt = 1.0,
    observation_strategy=SectionedStateObservation(minisections=32))
policy = StepwiseRDEPolicy(env, [20.0f0, 100.0f0, 200.0f0, 300.0f0, 400.0f0, 500.0f0, 600.0f0, 700.0f0], 
[0.64f0, 0.96f0, 0.45f0, 0.84f0, 0.5f0, 0.75f0, 0.4f0, 0.62f0])
## Policy used in training
policy = policies[end]
env = envs[end]
##
fig = compare_to_policy(;fno, ps, st, policy, env, cdev, gdev)
##
save(plotsdir("fno_test_modes16.png"), fig)
##
# Calculate test loss
test_loss = mean(abs2, output_data .- test_data[:,:,2:end])
@info "Test Loss: $test_loss"

