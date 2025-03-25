using DrWatson
@quickactivate :RDEML

## Setup devices
rng = Random.default_rng()
const cdev = cpu_device()
const gdev = gpu_device()

## Load data
train_data, test_data = prepare_dataset(;batch_size=512, test_split=0.2, rng, create_loader=false)
train_loader = create_dataloader(train_data, batch_size=512)
test_loader = create_dataloader(test_data, batch_size=min(2^12, number_of_samples(test_data)))
## Setup FNO, and train it
fno_config = FNOConfig(;rng)
fno_config.ps = fno_config.ps |> gdev
fno_config.st = fno_config.st |> gdev
@elapsed test_loss = RDEML.evaluate_test_loss(FNO(fno_config), fno_config.ps, fno_config.st, test_loader, gdev)

@time train!(fno_config, train_loader, [0.01f0, 0.001f0, 3f-4], [3, 2, 1]; dev=gdev, test_data=test_loader)
plot_losses(fno_config; saveplot=false)
length(fno_config.history.losses) / sum(fno_config.history.epochs)

## More training
@time train!(fno_config, data, 0.001f0, 10; dev=gdev)
plot_losses(fno_config)
@time train!(fno_config, data, 3f-4, 20; dev=gdev)
plot_losses(fno_config)

## Save the model
safesave(datadir("fno", "first_after_data_fix.jld2"), fno_config)
wload(datadir("fno", "first_after_data_fix.jld2"))
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
