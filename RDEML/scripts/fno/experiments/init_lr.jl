using DrWatson
@quickactivate :RDEML
##
experiment_name = "init_lr_2"
batch_size = 256
modes = 64
width = 32
depth = 4
lrs = [0.1f0, 0.01f0, 0.001f0, 0.0001f0]
epochs = 20
runs_per_config = 6
## rng and devices
experiment_seed = 477
rng = Random.default_rng()
Random.seed!(rng, experiment_seed)
const cdev = cpu_device()
const gdev = gpu_device()
## Load data with train/test split
train_dataset, test_dataset = prepare_dataset(create_loader=false, test_split=0.2, rng=rng)

test_loader = create_dataloader(test_dataset; batch_size=2^12, shuffle=false)
train_loader = create_dataloader(train_dataset; batch_size, shuffle=true)
## Run experiments
for (i, lr) in enumerate(lrs)
    for run_id in 1:runs_per_config
        cfg = Dict("init_lr" => lr, "run" => run_id)
        produce_or_load(cfg, datadir("fno", experiment_name)) do cfg
            # Reseed with a unique seed for this run
            Random.seed!(rng, experiment_seed + (i-1)*runs_per_config + run_id)
            
            chs = (3, fill(width, depth)..., 2)
            config = FNOConfig(;chs, modes, rng)
            
            # Use train_loader and test_loader
            train!(config, train_loader, [lr], [epochs]; test_data=test_loader, dev=gdev)
            
            # Move parameters and state to CPU
            config.ps = config.ps |> cdev
            config.st = config.st |> cdev
            # Include test loss in results
            result = Dict(
                "init_lr" => lr, 
                "run" => run_id, 
                "full_config" => config, 
                "final_train_loss" => config.history.losses[end],
                "final_test_loss" => config.history.test_losses[end]
            )
            return result
        end
    end
end

