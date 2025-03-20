using DrWatson
@quickactivate "RDE_PINNs"
using DRL_RDE_utils
using RDEML
using Random
using Lux
using LuxCUDA
## Get Policies and evironments for collecting data
policies, envs = make_data_policies_and_envs()
reset_strategies = make_data_reset_strategies()
## Collect data
n_runs_per_reset_strategy = 4
run_data, data = collect_data(policies, envs, reset_strategies; n_runs_per_reset_strategy)
save_data(run_data, data; filename="RDE_sim_data_2.jld2")
## Visualize data
visualize_data(run_data, policies, envs, reset_strategies, n_runs_per_reset_strategy; save_plots=true)
## Load data
all_data = load(datadir("RDE_sim_data_2.jld2"))
run_data = all_data["run_data"]
data = all_data["data"]
## Setup devices
rng = Random.default_rng()
const cdev = cpu_device()
const gdev = gpu_device(2)
## Setup FNO
fno_config = FNOConfig()
@time train!(fno_config, data, 0.01f0, 10; dev=gdev)
plot_losses(fno_config; saveplot=true)
##more training
@time train!(fno_config, data, 0.001f0, 50; dev=gdev)
@time train!(fno_config, data, 3f-5, 100; dev=gdev)
plot_losses(fno_config; saveplot=true)
## Save the model
safesave(datadir("fno", savename(fno_config, "jld2")), fno_config)






##
RDEParams = RDEParam(tmax = 410.0)
env = RDEEnv(RDEParams,
    dt = 1.0,
    observation_strategy=SectionedStateObservation(minisections=32), 
    reset_strategy=WeightedCombination(Float32[0.15, 0.5, 0.2, 0.15]))
policy = StepwiseRDEPolicy(env, [20.0f0, 100.0f0, 200.0f0, 300.0f0, 400.0f0, 500.0f0, 600.0f0, 700.0f0], 
[0.64f0, 0.96f0, 0.45f0, 0.7f0, 0.5f0, 0.75f0, 0.4f0, 0.55f0])
## Policy used in training
policy = policies[end]
env = envs[end]
##
fig = compare_to_policy(;fnoconfig=fno_config, policy, env, cdev, gdev, timesteps=[1, 10, 20])
fig = compare_to_policy(;fnoconfig=fno_config, policy, env, cdev, gdev, recursive=true, timesteps=[1, 10, 20])