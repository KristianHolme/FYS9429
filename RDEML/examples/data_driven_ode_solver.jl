using DrWatson
@quickactivate :RDEML

## Setup devices
rng = Random.default_rng()
const cdev = cpu_device()
const gdev = gpu_device()

## Load data
train_loader, test_loader = prepare_dataset(;batch_size=256, test_split=0.2, rng, create_loader=true)
## Setup FNO, and train it
fno_config = FNOConfig(;rng)
fno_config.ps = fno_config.ps |> gdev
fno_config.st = fno_config.st |> gdev
# @elapsed test_loss = RDEML.evaluate_test_loss(FNO(fno_config), fno_config.ps, fno_config.st, test_loader, gdev)

lrs = 10 .^LinRange(-2, -5, 15)
epochs = fill(2, length(lrs))
@time training_state = train!(fno_config, train_loader, lrs, epochs; dev=gdev, test_data=test_loader)
plot_losses(fno_config; saveplot=false)




## Save the model
# transfer parameters and state to CPU
fno_config.ps = fno_config.ps |> cdev
fno_config.st = fno_config.st |> cdev
# save the model
safesave(datadir("fno", "first_after_data_fix.jld2"), fno_config)
# Load the model
fno_config = wload(datadir("fno", "first_after_data_fix.jld2"))
