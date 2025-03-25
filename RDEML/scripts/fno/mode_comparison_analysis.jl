## Plotting
using DrWatson
@quickactivate :RDEML
using CairoMakie
using Statistics
##
experiment_name = "mode_comparison"
dataset = prepare_dataset(create_loader=false)
n_data = number_of_samples(dataset)

df = collect_results(datadir("fno", experiment_name))
fno_configs = [wload(row.path, FNOConfig) for row in eachrow(df)]

modes_list = sort(unique(df.modes)) |> Vector{Int}
##
fig = Figure(size=(1000, 600))

# First plot - Loss history
ax1 = Axis(fig[1, 1], xlabel="Epoch", ylabel="Loss (moving average)", yscale=log10)

for (i, row) in enumerate(eachrow(df))
    config = row.full_config
    losses = config.history.losses
    n_batches = calculate_number_of_batches(n_data, row.batch_size)
    training_steps = 1:length(losses) |> collect
    total_epochs = sum(config.history.epochs)
    epochs = training_steps ./ n_batches
    
    # Get color for this batch size
    modes = row.modes
    color_idx = indexin(modes, modes_list)[1]
    color = Makie.wong_colors()[color_idx]
    
    # Plot line
    line = lines!(ax1, epochs, moving_average(losses, Int(floor(length(losses) / 100))), color=color)
end

# Second plot - Individual end loss barplot
ax2 = Axis(fig[1, 2], xlabel="Modes", ylabel="Final Loss", yscale=log10,
    xticks = (1:length(modes_list), string.(modes_list)))

# Collect final losses and batch sizes
final_losses = df.final_loss |> collect |> Vector{Float32}
modes = df.modes |> collect |> Vector{Int}
group = [indexin(modes, modes_list)[1] for modes in modes]
run_id = df.run |> collect |> Vector{Int}

wcolors = Makie.wong_colors()[1:length(modes_list)]
colors = wcolors[group]
barplot!(ax2, group, final_losses, 
    dodge=run_id, color = colors)
fig
##
save(plotsdir("fno", experiment_name, "mode_comparison_analysis.svg"), fig)
save(plotsdir("fno", experiment_name, "mode_comparison_analysis.png"), fig)
