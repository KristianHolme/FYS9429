using DrWatson
@quickactivate :RDEML
##
experiment_name = "width_comparison_2"
widths_list = [16, 32, 64, 128]
lr_list = [0.01f0, 0.001f0, 3f-4]
epochs_list = [15, 10, 5]
runs_per_config = 6
## rng and devices
rng = Random.default_rng()
experiment_seed = 942
Random.seed!(rng, experiment_seed)
const cdev = cpu_device()
const gdev = gpu_device()
## Load data
train_loader, test_loader = prepare_dataset(;rng, batch_size=256, test_split=0.2)

## Run experiments
for width in widths_list
    for run_id in 1:runs_per_config
        cfg = Dict("width" => width, "run" => run_id)
        produce_or_load(cfg, datadir("fno", experiment_name)) do cfg
            Random.seed!(rng, experiment_seed + run_id)
            chs = (3, width, width, width, 2)
            modes = 64
            config = FNOConfig(;chs, modes, rng)
            train!(config, train_loader, lr_list, epochs_list; dev=gdev, test_data=test_loader)
            config.ps = config.ps |> cdev
            config.st = config.st |> cdev
            result = Dict("width" => width, "run" => run_id, "full_config" => config,
                "final_train_loss" => config.history.losses[end],
                "final_test_loss" => config.history.test_losses[end])
            return result
        end
    end
end

