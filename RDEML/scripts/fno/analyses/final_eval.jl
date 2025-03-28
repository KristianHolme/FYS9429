using DrWatson
@quickactivate :RDEML
## Test using old






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