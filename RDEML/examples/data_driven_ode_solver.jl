using DrWatson
@quickactivate :RDEML
##
using Random
using Lux
using LuxCUDA
## Load data
data = prepare_dataset("test")

## Setup devices
rng = Random.default_rng()
const cdev = cpu_device()
const gdev = gpu_device(2)

## Setup FNO, and train it
fno_config = FNOConfig()
@time train!(fno_config, data, 0.01f0, 10; dev=gdev)
plot_losses(fno_config; saveplot=true)

## More training
@time train!(fno_config, data, 0.001f0, 50; dev=gdev)
@time train!(fno_config, data, 3f-5, 100; dev=gdev)
plot_losses(fno_config; saveplot=true)

## Save the model
safesave(datadir("fno", savename(fno_config, "jld2")), fno_config)

## Test FNO with Policy used in training
policies, envs = make_data_policies_and_envs()
reset_strategies = make_data_reset_strategies()
policy = policies[end]
env = envs[end]
env.prob.reset_strategy = reset_strategies[end]
# plot selected predictions
fig = compare_to_policy(;fnoconfig=fno_config, policy, env, cdev, gdev, timesteps=[1, 10, 20])
fig = compare_to_policy(;fnoconfig=fno_config, policy, env, cdev, gdev, recursive=true, timesteps=[1, 10, 20])

## Test the FNO with a (probably) new initial condition
RDEParams = RDEParam(tmax = 410.0)
env = RDEEnv(RDEParams,
    dt = 1.0,
    observation_strategy=SectionedStateObservation(minisections=32), 
    reset_strategy=WeightedCombination(Float32[0.15, 0.5, 0.2, 0.15]))
policy = StepwiseRDEPolicy(env, [20.0f0, 100.0f0, 200.0f0, 300.0f0, 400.0f0], 
[0.64f0, 0.96f0, 0.45f0, 0.7f0, 0.5f0])
# plot selected predictions
fig = compare_to_policy(;fnoconfig=fno_config, policy, env, cdev, gdev, timesteps=[1, 10, 20])
fig = compare_to_policy(;fnoconfig=fno_config, policy, env, cdev, gdev, recursive=true, timesteps=[1, 10, 20])

## Test with different initial condition
env.prob.reset_strategy = SineCombination()
# plot selected predictions
fig = compare_to_policy(;fnoconfig=fno_config, policy, env, cdev, gdev, timesteps=[1, 10, 20])
fig = compare_to_policy(;fnoconfig=fno_config, policy, env, cdev, gdev, recursive=true, timesteps=[1, 10, 20])
