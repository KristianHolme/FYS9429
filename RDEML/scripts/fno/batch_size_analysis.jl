## Plotting
using DrWatson
@quickactivate :RDEML
using CairoMakie
using Statistics
##
experiment_name = "batch_size"

df = collect_results(datadir("fno", experiment_name));
fno_configs = [wload(row.path, FNOConfig) for row in eachrow(df)]

batch_size_list = sort(unique(df.batch_size)) |> Vector{Int}
##
fig = Figure(size=(1000, 600))

# First plot - Loss history
ax1 = Axis(fig[1, 1], xlabel="Epoch", ylabel="Loss (moving average)", yscale=log10)
for (i, row) in enumerate(eachrow(df))
    config = row.full_config
    losses = config.history.losses
    total_epochs = sum(config.history.epochs)
    n_batches = length(losses) ÷ total_epochs
    training_progress = 1:length(losses) |> collect ./ length(losses)
    epochs = training_progress .* total_epochs
    
    # Get color for this batch size
    batch_size = row.batch_size
    color_idx = indexin(batch_size, batch_size_list)[1]
    color = Makie.wong_colors()[color_idx]
    
    # Plot line
    line = lines!(ax1, epochs, moving_average(losses, Int(floor(length(losses) / 100))), color=color)
    # plot test loss
    test_losses = config.history.test_losses
    lines!(ax1, 1:total_epochs, test_losses, color=color, linestyle=:dash)
end
ax1_lines = [Linesegment(color=:black, linestyle=:solid), Linesegment(color=:black, linestyle=:dash)]
axislegend(ax1, ax1_lines, ["Train Loss", "Test Loss"], position=:tr)

# Second plot - Individual end loss barplot
ax2 = Axis(fig[1, 2], xlabel="Batch Size", ylabel="Final Loss", yscale=log10,
    xticks = (1:length(batch_size_list), string.(batch_size_list)))

# Collect final losses and batch sizes
final_losses = df.final_loss |> collect |> Vector{Float32}
batch_sizes = df.batch_size |> collect |> Vector{Int}
group = [indexin(batch_size, batch_size_list)[1] for batch_size in batch_sizes]
run_id = df.run |> collect |> Vector{Int}

wcolors = Makie.wong_colors()[1:length(batch_size_list)]
colors = wcolors[group]
barplot!(ax2, group, final_losses, 
    dodge=run_id, color = colors)
fig
##
save(plotsdir("fno", experiment_name, "batch_size_analysis.svg"), fig)
save(plotsdir("fno", experiment_name, "batch_size_analysis.png"), fig)
