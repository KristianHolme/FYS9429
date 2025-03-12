# Example script for using the RDEPINN framework to solve Rotating Detonation Engine equations

using DrWatson
@quickactivate "RDE_PINNs"

# Load the RDEPINN module once and don't use explicit module qualification
using RDEPINN                 # This imports all exported symbols

using CairoMakie
using Random

# Set random seed for reproducibility
Random.seed!(123)

@info "RDEPINN Example: Solving Rotating Detonation Engine equations with Physics-Informed Neural Networks"
##
# Create configurations
@info "Creating configurations..."
pde_config = default_pde_config(tmax=1.0, u_scale=1.5)
model_config = default_model_config(hidden_sizes=[32, 32, 32])
training_config = fast_training_config()

# Display configurations
display(pde_config)
display(model_config)
display(training_config)

# Create and run a basic experiment
@info "Running basic experiment..."
setup = create_experiment("basic_rde", pde_config, model_config, training_config)
result_dict = run_experiment(setup)

# Plots are automatically saved by the run_experiment function
@info "Plots have been saved to $(plotsdir("experiments", setup.name))"

## Run a hyperparameter sweep on hidden layer sizes
@info "Running hyperparameter sweep on hidden layer sizes..."
hidden_sizes_values = [[16, 16], ones(Int64, 16)*16, [64, 64]]
result_dicts, comparison = run_hyperparameter_sweep(
    setup, 
    "hidden_sizes", 
    hidden_sizes_values, 
    experiment_name_prefix="hidden_size_sweep"
)

@info "Hyperparameter sweep results:"
for (i, dict) in enumerate(result_dicts)
    exp_display = experiment_display(dict)
    @info "Model $(i)" hidden_sizes=exp_display.setup.model_config.hidden_sizes mse=round(exp_display.results.metrics.mse, digits=6) final_loss=round(exp_display.results.metrics.final_loss, digits=6)
end

# Find the best model based on final loss
best_idx = argmin([dict["metrics"].final_loss for dict in result_dicts])
best_dict = result_dicts[best_idx]
best_display = experiment_display(best_dict)

@info "Best model found" hidden_sizes=best_display.setup.model_config.hidden_sizes mse=round(best_display.results.metrics.mse, digits=6) final_loss=round(best_display.results.metrics.final_loss, digits=6)

@info "Example completed successfully!"
@info "All results have been saved to:" experiments=datadir("experiments") plots=plotsdir("experiments") sweeps=datadir("sweeps") 