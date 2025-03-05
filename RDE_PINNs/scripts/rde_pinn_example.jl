# Example script for using the RDEPINN framework to solve Rotating Detonation Engine equations

using DrWatson
@quickactivate "RDE_PINNs"

# Load the RDEPINN module once and don't use explicit module qualification
using RDEPINN                 # This imports all exported symbols

using CairoMakie
using Random

# Set random seed for reproducibility
Random.seed!(123)

println("RDEPINN Example: Solving Rotating Detonation Engine equations with Physics-Informed Neural Networks")
println("=========================================================================================")

# Create configurations
println("Creating configurations...")
pde_config = default_pde_config(tmax=1.0, u_scale=1.5)
model_config = default_model_config(hidden_sizes=[32, 32, 32])
training_config = fast_training_config()

# Print configurations
println("PDE Configuration: tmax=$(pde_config.rde_params.tmax), u_scale=$(pde_config.u_scale)")
println("Model Configuration: hidden_sizes=$(model_config.hidden_sizes)")
println("Training Configuration: iterations=$(training_config.iterations), batch_size=$(training_config.batch_size)")
println()

# Create and run a basic experiment
println("Running basic experiment...")
experiment = create_experiment("basic_rde", pde_config, model_config, training_config)
experiment = run_experiment(experiment)

# Print experiment summary
println("\nExperiment Summary:")
print_experiment_summary(experiment)

# Plots are automatically saved by the run_experiment function
println("\nPlots have been saved to $(plotsdir("experiments", experiment.name))")

# Run a hyperparameter sweep on hidden layer sizes
println("\nRunning hyperparameter sweep on hidden layer sizes...")
hidden_sizes_values = [[16, 16], [32, 32, 32], [64, 64, 64, 64]]
experiments, comparison = run_hyperparameter_sweep(
    experiment, 
    "hidden_sizes", 
    hidden_sizes_values, 
    experiment_name_prefix="hidden_size_sweep"
)

println("\nHyperparameter sweep results:")
for (i, exp) in enumerate(experiments)
    println("Model with hidden_sizes=$(exp.model_config.hidden_sizes):")
    println("  MSE: $(round(exp.metrics.mse, digits=6))")
    println("  Final Loss: $(round(exp.metrics.final_loss, digits=6))")
end

# Find the best model based on final loss
best_idx = argmin([exp.metrics.final_loss for exp in experiments])
best_experiment = experiments[best_idx]
println("\nBest model: hidden_sizes=$(best_experiment.model_config.hidden_sizes)")
println("  MSE: $(round(best_experiment.metrics.mse, digits=6))")
println("  Final Loss: $(round(best_experiment.metrics.final_loss, digits=6))")

println("\nExample completed successfully!")
println("All results have been saved to:")
println("  - Experiments: $(datadir("experiments"))")
println("  - Plots: $(plotsdir("experiments"))")
println("  - Sweeps: $(datadir("sweeps"))") 