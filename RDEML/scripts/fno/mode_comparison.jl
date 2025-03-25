using DrWatson
@quickactivate :RDEML
##
experiment_name = "mode_comparison"
modes_list = [8, 16, 32, 64]
lr_list = [0.01f0, 0.001f0, 3f-4]
epochs_list = [15, 10, 5]
runs_per_config = 6
## rng and devices
rng = Random.default_rng()
Random.seed!(rng, 963)
const cdev = cpu_device()
const gdev = gpu_device(1)
## Load data
train_loader, test_loader = prepare_dataset(;rng, batch_size=512, test_split=0.2)

## Run experiments
for modes in modes_list
    for run_id in 1:runs_per_config
        cfg = Dict("modes" => modes, "run" => run_id)
        produce_or_load(cfg, datadir("fno", experiment_name)) do cfg
            config = FNOConfig(;modes, rng)
            train!(config, train_loader, lr_list, epochs_list; dev=gdev, test_data=test_loader)
            config.ps = config.ps |> cdev
            config.st = config.st |> cdev
            result = Dict("modes" => modes, "run" => run_id, "full_config" => config,
                "final_loss" => config.history.losses[end],
                "final_test_loss" => config.history.test_losses[end])
            return result
        end
    end
end

