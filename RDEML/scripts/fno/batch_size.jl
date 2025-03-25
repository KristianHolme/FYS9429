using DrWatson
@quickactivate :RDEML
##
experiment_name = "batch_size"
batch_size_list = [128, 256, 512, 1024, 2048, 4096]
lr_list = [0.01f0, 0.001f0, 3f-4]
epochs_list = [15, 10, 5]
runs_per_config = 6
## rng and devices
rng = Random.default_rng()
Random.seed!(rng, 403)
const cdev = cpu_device()
const gdev = gpu_device()
## Load data with train/test split
train_dataset, test_dataset = prepare_dataset(create_loader=false, test_split=0.2, rng=rng)

test_loader = create_dataloader(test_dataset; batch_size=number_of_samples(test_dataset), shuffle=false)
## Run experiments
for (i, batch_size) in enumerate(batch_size_list)
    # Create dataloaders with the current batch size
    train_loader = create_dataloader(train_dataset; batch_size, shuffle=true)
    
    for run_id in 1:runs_per_config
        cfg = Dict("batch_size" => batch_size, "run" => run_id)
        produce_or_load(cfg, datadir("fno", experiment_name)) do cfg
            config = FNOConfig(;rng)
            
            # Use train_loader and test_loader
            train!(config, train_loader, lr_list, epochs_list; test_data=test_loader, dev=gdev)
            
            # Move parameters and state to CPU
            config.ps = config.ps |> cdev
            config.st = config.st |> cdev
            # Include test loss in results
            result = Dict(
                "batch_size" => batch_size, 
                "run" => run_id, 
                "full_config" => config, 
                "final_train_loss" => config.history.losses[end],
                "final_test_loss" =>config.history.test_losses[end]
            )
            return result
        end
    end
end

