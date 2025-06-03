using DrWatson
@quickactivate :project_2
using DRiL.TensorBoardLogger
using ValueHistories
##
include("hyperParamSearch.jl")
##
experiment_name = "ppo_search_2025-06-01_21-27"
df, best_config = analyze_results(experiment_name, "Pendulum")

# Get the trial_id from the best config
best_trial_id = best_config.trial_id |> Int
println("Best trial ID: $best_trial_id")

# Scan logs directory for all seeds of this trial
logs_dir = datadir("experiments", "Pendulum", experiment_name, "logs")

# Find all seed directories for the best trial
seed_dirs = []
if isdir(logs_dir)
    for item in readdir(logs_dir)
        if startswith(item, "trial_$(best_trial_id)_seed_")
            push!(seed_dirs, item)
        end
    end
end

println("Found $(length(seed_dirs)) seed directories: $seed_dirs")

# Load TBReader for each seed and convert to ValueHistories
value_histories = []
for seed_dir in seed_dirs
    full_path = joinpath(logs_dir, seed_dir)
    println("Loading: $full_path")

    try
        reader = TBReader(full_path)

        # Convert TBReader to ValueHistories.MVHistory
        mv_history = convert(MVHistory, reader)

        push!(value_histories, mv_history)
        println("  ✓ Successfully loaded and converted $seed_dir")

    catch e
        println("  ✗ Failed to load $seed_dir: $e")
    end
end

println("\nData gathering complete!")
println("Loaded $(length(value_histories)) ValueHistory objects")

# Show what metrics are available
if !isempty(value_histories)
    available_metrics = keys(value_histories[1])
    println("Available metrics: $available_metrics")
end

value_histories[1][Symbol("env/ep_rew_mean")] |> get