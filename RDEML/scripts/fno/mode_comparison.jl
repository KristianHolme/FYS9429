using DrWatson
@quickactivate :RDEML
##
experiment_name = "mode_comparison"
modes_list = [8, 16, 32, 64]
lr_list = [0.01f0, 0.001f0, 3f-4]
epochs_list = [50, 70, 100]
## rng and devices
rng = Random.default_rng()
Random.seed!(rng, 403)
const cdev = cpu_device()
const gdev = gpu_device(2)
## Load data
data = prepare_dataset(;rng)

## Run experiments
for (i, modes) in enumerate(modes_list)
    config = FNOConfig(;modes, rng)
    train_and_save!(config, data, lr_list, epochs_list; dev=gdev, folder=experiment_name)
    plot_losses(config; saveplot=true, folder=experiment_name)
end

