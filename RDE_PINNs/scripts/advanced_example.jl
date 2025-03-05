# Advanced example script for using the RDEPINN framework to solve Rotating Detonation Engine equations

using RDEPINN
using CairoMakie
using Random

# Set random seed for reproducibility
Random.seed!(123)

println("RDEPINN Advanced Example: Solving Rotating Detonation Engine equations with Physics-Informed Neural Networks")
println("=================================================================================================")

# 1. Compare different neural network architectures
println("\n1. Comparing different neural network architectures")
println("---------------------------------------------------")

# Create base configurations
pde_config = default_pde_config(tmax=1.0, u_scale=1.5)
training_config = medium_training_config()

# Define different model configurations to compare
model_configs = [
    small_model_config(),
    medium_model_config(),
    large_model_config(),
    ModelConfig([32, 64, 32], tanh, :glorot_uniform),
    ModelConfig([16, 32, 64, 32, 16], tanh, :glorot_uniform)
]

model_names = [
    "small",
    "medium",
    "large",
    "pyramid",
    "hourglass"
]

# Run experiments for each model configuration
model_experiments = []
for (i, (config, name)) in enumerate(zip(model_configs, model_names))
    println("Running experiment $i/$(length(model_configs)): $name architecture")
    experiment = create_experiment("architecture_$name", pde_config, config, training_config)
    experiment = run_experiment(experiment)
    save_experiment(experiment)
    push!(model_experiments, experiment)
end

# Compare model architectures
println("\nComparing model architectures:")
comparison = compare_experiments(model_experiments, metrics=["mse", "final_loss", "rmse"])

# Find the best model based on MSE
best_model_idx = argmin([exp.metrics.mse for exp in model_experiments])
best_model_experiment = model_experiments[best_model_idx]
println("\nBest model architecture: $(model_names[best_model_idx])")
println("  MSE: $(round(best_model_experiment.metrics.mse, digits=6))")
println("  RMSE: $(round(best_model_experiment.metrics.rmse, digits=6))")
println("  Final Loss: $(round(best_model_experiment.metrics.final_loss, digits=6))")

# Save comparison figure
fig = plot_experiment_comparison(
    model_experiments,
    metric="mse",
    title="RDE Model Architecture Comparison - MSE"
)
save("architecture_comparison_mse.png", fig)

fig = plot_experiment_comparison(
    model_experiments,
    metric="final_loss",
    title="RDE Model Architecture Comparison - Final Loss"
)
save("architecture_comparison_loss.png", fig)

# 2. Investigate effect of different initial conditions (u_scale)
println("\n2. Investigating effect of different initial conditions (u_scale)")
println("--------------------------------------------------------------")

# Use the best model configuration from the previous comparison
best_model_config = best_model_experiment.model_config

# Define different u_scale values to test
u_scale_values = [1.0, 1.25, 1.5, 1.75, 2.0]

# Run experiments for each u_scale value
u_scale_experiments = []
for (i, u_scale) in enumerate(u_scale_values)
    println("Running experiment $i/$(length(u_scale_values)): u_scale = $u_scale")
    current_pde_config = default_pde_config(tmax=1.0, u_scale=u_scale)
    experiment = create_experiment("u_scale_$(u_scale)", current_pde_config, best_model_config, training_config)
    experiment = run_experiment(experiment)
    save_experiment(experiment)
    push!(u_scale_experiments, experiment)
end

# Compare u_scale experiments
println("\nComparing u_scale values:")
comparison = compare_experiments(u_scale_experiments, metrics=["mse", "final_loss", "rmse"])

# Find the best u_scale based on MSE
best_u_scale_idx = argmin([exp.metrics.mse for exp in u_scale_experiments])
best_u_scale_experiment = u_scale_experiments[best_u_scale_idx]
best_u_scale = u_scale_values[best_u_scale_idx]
println("\nBest u_scale value: $(best_u_scale)")
println("  MSE: $(round(best_u_scale_experiment.metrics.mse, digits=6))")
println("  RMSE: $(round(best_u_scale_experiment.metrics.rmse, digits=6))")
println("  Final Loss: $(round(best_u_scale_experiment.metrics.final_loss, digits=6))")

# Save comparison figure
fig = plot_experiment_comparison(
    u_scale_experiments,
    metric="mse",
    title="RDE u_scale Comparison - MSE"
)
save("u_scale_comparison_mse.png", fig)

# 3. Create detailed spacetime slice plots for the best model
println("\n3. Creating detailed spacetime slice plots for the best model")
println("----------------------------------------------------------")

# Use the best model from the u_scale comparison
best_experiment = best_u_scale_experiment

# Create spacetime slices at different time points
time_points = [0.1, 0.3, 0.5, 0.7, 0.9]  # Normalized time points
ts = best_experiment.predictions[:ts]
xs = best_experiment.predictions[:xs]
us = best_experiment.predictions[:us]
λs = best_experiment.predictions[:λs]

for (i, t_norm) in enumerate(time_points)
    t_idx = max(1, min(length(ts), Int(round(t_norm * length(ts)))))
    t_val = ts[t_idx]
    
    println("Creating spacetime slice at t = $t_val (index $t_idx)")
    
    fig = plot_spacetime_slice(
        ts, xs, us, λs, t_idx,
        title="RDE Spacetime Slice at t = $t_val"
    )
    save("spacetime_slice_t$(i).png", fig)
end

# Create a multi-panel figure with all slices
fig = Figure(size=(1200, 800))
for (i, t_norm) in enumerate(time_points)
    t_idx = max(1, min(length(ts), Int(round(t_norm * length(ts)))))
    t_val = ts[t_idx]
    
    # Plot u
    ax1 = Axis(fig[i, 1], title="u(t=$t_val,x)", xlabel="x", ylabel="u")
    lines!(ax1, xs, us[t_idx], linewidth=2, color=:blue)
    
    # Plot λ
    ax2 = Axis(fig[i, 2], title="λ(t=$t_val,x)", xlabel="x", ylabel="λ")
    lines!(ax2, xs, λs[t_idx], linewidth=2, color=:red)
end

Label(fig[0, 1:2], "RDE Spacetime Slices at Different Time Points", fontsize=20)
save("spacetime_slices_combined.png", fig)

# Create animation with higher framerate and resolution
println("Creating high-quality animation for the best model...")
animation_file = create_animation(
    ts, xs, us, λs,
    filename="best_model_animation.mp4",
    framerate=30
)
println("Saved high-quality animation to $animation_file")

# 4. Analyze metrics in detail
println("\n4. Analyzing metrics in detail")
println("-----------------------------")

# Plot metrics over time for the best model
println("Plotting metrics over time...")
fig = plot_metrics_over_time(
    best_experiment,
    metrics=["loss"],
    title="RDE Training Loss Over Time"
)
save("best_model_loss.png", fig)

# Create a detailed error analysis
println("Creating detailed error analysis...")
fig = plot_error(
    best_experiment.predictions[:ts],
    best_experiment.predictions[:xs],
    best_experiment.predictions[:us],
    best_experiment.predictions[:λs],
    best_experiment.predictions[:us_sim],
    best_experiment.predictions[:λs_sim],
    title="RDE Detailed Error Analysis"
)
save("best_model_error.png", fig)

# 5. Experiment with different training strategies
println("\n5. Experimenting with different training strategies")
println("------------------------------------------------")

# Define different training configurations
training_configs = [
    TrainingConfig(Adam(0.01), 1000, 100, 100, 100),   # Default
    TrainingConfig(Adam(0.001), 2000, 100, 100, 100),  # Lower learning rate, more iterations
    TrainingConfig(Adam(0.05), 500, 100, 100, 100),    # Higher learning rate, fewer iterations
    TrainingConfig(Adam(0.01), 1000, 50, 100, 100),    # Smaller batch size
    TrainingConfig(Adam(0.01), 1000, 200, 100, 100)    # Larger batch size
]

training_names = [
    "default",
    "low_lr_long",
    "high_lr_short",
    "small_batch",
    "large_batch"
]

# Run experiments for each training configuration
training_experiments = []
for (i, (config, name)) in enumerate(zip(training_configs, training_names))
    println("Running experiment $i/$(length(training_configs)): $name training strategy")
    experiment = create_experiment(
        "training_$name",
        best_experiment.pde_config,
        best_experiment.model_config,
        config
    )
    experiment = run_experiment(experiment)
    save_experiment(experiment)
    push!(training_experiments, experiment)
end

# Compare training strategies
println("\nComparing training strategies:")
comparison = compare_experiments(training_experiments, metrics=["mse", "final_loss", "rmse"])

# Find the best training strategy based on MSE
best_training_idx = argmin([exp.metrics.mse for exp in training_experiments])
best_training_experiment = training_experiments[best_training_idx]
println("\nBest training strategy: $(training_names[best_training_idx])")
println("  MSE: $(round(best_training_experiment.metrics.mse, digits=6))")
println("  RMSE: $(round(best_training_experiment.metrics.rmse, digits=6))")
println("  Final Loss: $(round(best_training_experiment.metrics.final_loss, digits=6))")

# Save comparison figure
fig = plot_experiment_comparison(
    training_experiments,
    metric="mse",
    title="RDE Training Strategy Comparison - MSE"
)
save("training_comparison_mse.png", fig)

# Plot loss curves for all training strategies
fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1], title="RDE Training Loss Comparison", xlabel="Iteration", ylabel="Loss")

for (i, exp) in enumerate(training_experiments)
    if haskey(exp.metrics.history, :loss)
        values = exp.metrics.history[:loss]
        iterations = 1:length(values)
        lines!(ax, iterations, values, label=training_names[i])
    end
end

axislegend(ax)
save("training_loss_comparison.png", fig)

println("\nAdvanced example completed successfully!")
println("Check the generated files and experiment directories for results.") 