using DrWatson
@quickactivate :RDEML
##
experiment_name = "final_runs_2"
depth = 4
width = 32
modes = 16
batch_size = 512
lrs = [0.01f0, 0.001f0, 3f-4]
epochs = [5, 10, 10]
runs = 24
experiment_seed = 869
## Setup rng and devices
rng = Random.default_rng()
Random.seed!(rng, experiment_seed)
const cdev = cpu_device()
const gdev = gpu_device()
## Load data
train_loader, test_loader = prepare_dataset(;rng, batch_size, test_split=0.2)

## Run experiments
for run in 1:runs
    cfg = Dict("run" => run)
    produce_or_load(cfg, datadir("fno", experiment_name)) do cfg
        Random.seed!(rng, experiment_seed + run)
        chs = (3, (Int.(ones(depth).* width))..., 2)
        config = FNOConfig(;chs, modes, rng)
        train!(config, train_loader, lrs, epochs; dev=gdev, test_data=test_loader)
        config.ps = config.ps |> cdev
        config.st = config.st |> cdev
        result = Dict("run" => run, "full_config" => config,
            "final_train_loss" => config.history.losses[end],
            "final_test_loss" => config.history.test_losses[end])
        return result
    end
end