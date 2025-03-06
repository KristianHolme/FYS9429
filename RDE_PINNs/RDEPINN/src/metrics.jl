"""
    Metrics

Structure to store metrics for evaluating Rotating Detonation Engine (RDE) PINN models.
"""
struct Metrics
    mse::Float64
    mae::Float64
    rmse::Float64
    r_squared::Float64
    final_loss::Float64
    training_time::Float64  # in seconds
    history::Dict{Symbol, Vector{Float64}}
end

"""
    calculate_mse(predictions, targets)

Calculate Mean Squared Error for RDE model predictions.
"""
function calculate_mse(predictions, targets)
    return mean((predictions .- targets).^2)
end

"""
    calculate_mae(predictions, targets)

Calculate Mean Absolute Error for RDE model predictions.
"""
function calculate_mae(predictions, targets)
    return mean(abs.(predictions .- targets))
end

"""
    calculate_rmse(predictions, targets)

Calculate Root Mean Squared Error for RDE model predictions.
"""
function calculate_rmse(predictions, targets)
    return sqrt(calculate_mse(predictions, targets))
end

"""
    calculate_r_squared(predictions, targets)

Calculate R² (coefficient of determination) for RDE model predictions.
"""
function calculate_r_squared(predictions, targets)
    ss_total = sum((targets .- mean(targets)).^2)
    ss_residual = sum((targets .- predictions).^2)
    return 1 - ss_residual / ss_total
end

"""
    calculate_metrics(predictions, targets, final_loss, loss_history, training_time=0.0)

Calculate all metrics for a Rotating Detonation Engine PINN model.
"""
function calculate_metrics(predictions, targets, final_loss, loss_history, training_time=0.0)
    mse = calculate_mse(predictions, targets)
    mae = calculate_mae(predictions, targets)
    rmse = calculate_rmse(predictions, targets)
    r_squared = calculate_r_squared(predictions, targets)
    
    history = Dict{Symbol, Vector{Float64}}()
    history[:loss] = loss_history
    
    return Metrics(mse, mae, rmse, r_squared, final_loss, training_time, history)
end

"""
    compare_metrics(metrics_list, names)

Compare metrics from multiple Rotating Detonation Engine experiments.
"""
function compare_metrics(metrics_list, names)
    comparison = Dict{Symbol, Vector{Float64}}()
    
    # Extract each metric type
    comparison[:mse] = [m.mse for m in metrics_list]
    comparison[:mae] = [m.mae for m in metrics_list]
    comparison[:rmse] = [m.rmse for m in metrics_list]
    comparison[:r_squared] = [m.r_squared for m in metrics_list]
    comparison[:final_loss] = [m.final_loss for m in metrics_list]
    comparison[:training_time] = [m.training_time for m in metrics_list]
    
    # Create a table for display
    println("Metrics Comparison for RDE Experiments:")
    println("----------------------------------------")
    println("Experiment\tMSE\t\tMAE\t\tRMSE\t\tR²\t\tFinal Loss\tTraining Time (s)")
    
    for i in 1:length(names)
        println("$(names[i])\t$(round(comparison[:mse][i], digits=6))\t$(round(comparison[:mae][i], digits=6))\t$(round(comparison[:rmse][i], digits=6))\t$(round(comparison[:r_squared][i], digits=6))\t$(round(comparison[:final_loss][i], digits=6))\t$(round(comparison[:training_time][i], digits=2))")
    end
    
    return comparison
end

"""
    compute_mse(predicted, actual)

Compute the mean squared error between predicted and actual values.
"""
function compute_mse(predicted, actual)
    return mean((predicted .- actual).^2)
end

"""
    compute_mae(predicted, actual)

Compute the mean absolute error between predicted and actual values.
"""
function compute_mae(predicted, actual)
    return mean(abs.(predicted .- actual))
end

"""
    compute_rmse(predicted, actual)

Compute the root mean squared error between predicted and actual values.
"""
function compute_rmse(predicted, actual)
    return sqrt(compute_mse(predicted, actual))
end

"""
    compute_max_error(predicted, actual)

Compute the maximum absolute error between predicted and actual values.
"""
function compute_max_error(predicted, actual)
    return maximum(abs.(predicted .- actual))
end

"""
    compute_r2(predicted, actual)

Compute the R² score between predicted and actual values.
"""
function compute_r2(predicted, actual)
    ss_res = sum((predicted .- actual).^2)
    ss_tot = sum((actual .- mean(actual)).^2)
    return 1 - ss_res / ss_tot
end

"""
    compute_metrics(training_result::TrainingResult, simulation_data)

Compute metrics comparing the training result with simulation data.
"""
function compute_metrics(training_result::TrainingResult, simulation_data)
    ts_sim, xs_sim, us_sim, λs_sim = simulation_data
    
    # Interpolate neural network predictions to simulation grid if needed
    if length(training_result.ts) != length(ts_sim) || length(training_result.xs) != length(xs_sim)
        # Interpolate using the prediction functions
        us_interp = [training_result.predict_u.(t, xs_sim) for t in ts_sim]
        λs_interp = [training_result.predict_λ.(t, xs_sim) for t in ts_sim]
    else
        us_interp = training_result.us
        λs_interp = training_result.λs
    end
    
    # Flatten arrays for metric computation
    us_flat = vcat(us_interp...)
    λs_flat = vcat(λs_interp...)
    us_sim_flat = vcat(us_sim...)
    λs_sim_flat = vcat(λs_sim...)
    
    # Compute metrics
    metrics = Dict{String, Any}()
    
    # Overall metrics
    metrics["u_mse"] = compute_mse(us_flat, us_sim_flat)
    metrics["u_mae"] = compute_mae(us_flat, us_sim_flat)
    metrics["u_rmse"] = compute_rmse(us_flat, us_sim_flat)
    metrics["u_max_error"] = compute_max_error(us_flat, us_sim_flat)
    metrics["u_r2"] = compute_r2(us_flat, us_sim_flat)
    
    metrics["λ_mse"] = compute_mse(λs_flat, λs_sim_flat)
    metrics["λ_mae"] = compute_mae(λs_flat, λs_sim_flat)
    metrics["λ_rmse"] = compute_rmse(λs_flat, λs_sim_flat)
    metrics["λ_max_error"] = compute_max_error(λs_flat, λs_sim_flat)
    metrics["λ_r2"] = compute_r2(λs_flat, λs_sim_flat)
    
    # Time-specific metrics
    metrics["u_mse_by_time"] = [compute_mse(us_interp[i], us_sim[i]) for i in 1:length(ts_sim)]
    metrics["λ_mse_by_time"] = [compute_mse(λs_interp[i], λs_sim[i]) for i in 1:length(ts_sim)]
    
    # Final state metrics
    metrics["u_final_mse"] = compute_mse(us_interp[end], us_sim[end])
    metrics["λ_final_mse"] = compute_mse(λs_interp[end], λs_sim[end])
    
    return metrics
end

"""
    print_metrics(metrics::Dict{String, Any})

Print metrics in a formatted way.
"""
function print_metrics(metrics::Dict{String, Any})
    println("===== Evaluation Metrics =====")
    println("u variable:")
    println("  MSE: $(metrics["u_mse"])")
    println("  MAE: $(metrics["u_mae"])")
    println("  RMSE: $(metrics["u_rmse"])")
    println("  Max Error: $(metrics["u_max_error"])")
    println("  R²: $(metrics["u_r2"])")
    println()
    println("λ variable:")
    println("  MSE: $(metrics["λ_mse"])")
    println("  MAE: $(metrics["λ_mae"])")
    println("  RMSE: $(metrics["λ_rmse"])")
    println("  Max Error: $(metrics["λ_max_error"])")
    println("  R²: $(metrics["λ_r2"])")
    println()
    println("Final state:")
    println("  u MSE: $(metrics["u_final_mse"])")
    println("  λ MSE: $(metrics["λ_final_mse"])")
end

"""
    plot_metrics(metrics::Dict{String, Any}, ts_sim)

Plot metrics over time.
"""
function plot_metrics(metrics::Dict{String, Any}, ts_sim)
    fig = Figure(size=(800, 600))
    
    ax_mse = Axis(fig[1, 1], title="MSE over time", xlabel="t", ylabel="MSE")
    
    lines!(ax_mse, ts_sim, metrics["u_mse_by_time"], label="u MSE", linewidth=2)
    lines!(ax_mse, ts_sim, metrics["λ_mse_by_time"], label="λ MSE", linewidth=2)
    
    ax_mse.yscale = log10
    axislegend(ax_mse)
    
    return fig
end

"""
    compare_metrics(metrics_list::Vector{Dict{String, Any}}, labels::Vector{String})

Compare metrics from multiple models.
"""
function compare_metrics(metrics_list::Vector{Dict{String, Any}}, labels::Vector{String})
    if length(metrics_list) != length(labels)
        error("Number of metrics ($(length(metrics_list))) must match number of labels ($(length(labels)))")
    end
    
    fig = Figure(size=(1200, 800))
    
    # Compare u metrics
    ax_u = Axis(fig[1, 1], title="u metrics comparison", xlabel="Model", ylabel="Value")
    
    u_metrics = ["u_mse", "u_mae", "u_rmse"]
    u_values = [[metrics[metric] for metrics in metrics_list] for metric in u_metrics]
    
    barplot!(ax_u, 1:length(labels), u_values[1], label="MSE")
    barplot!(ax_u, (1:length(labels)) .+ 0.2, u_values[2], label="MAE")
    barplot!(ax_u, (1:length(labels)) .+ 0.4, u_values[3], label="RMSE")
    
    ax_u.xticks = (1:length(labels), labels)
    ax_u.yscale = log10
    axislegend(ax_u)
    
    # Compare λ metrics
    ax_λ = Axis(fig[1, 2], title="λ metrics comparison", xlabel="Model", ylabel="Value")
    
    λ_metrics = ["λ_mse", "λ_mae", "λ_rmse"]
    λ_values = [[metrics[metric] for metrics in metrics_list] for metric in λ_metrics]
    
    barplot!(ax_λ, 1:length(labels), λ_values[1], label="MSE")
    barplot!(ax_λ, (1:length(labels)) .+ 0.2, λ_values[2], label="MAE")
    barplot!(ax_λ, (1:length(labels)) .+ 0.4, λ_values[3], label="RMSE")
    
    ax_λ.xticks = (1:length(labels), labels)
    ax_λ.yscale = log10
    axislegend(ax_λ)
    
    # Compare R² scores
    ax_r2 = Axis(fig[2, :], title="R² scores", xlabel="Model", ylabel="R²")
    
    r2_metrics = ["u_r2", "λ_r2"]
    r2_values = [[metrics[metric] for metrics in metrics_list] for metric in r2_metrics]
    
    barplot!(ax_r2, 1:length(labels), r2_values[1], label="u R²")
    barplot!(ax_r2, (1:length(labels)) .+ 0.2, r2_values[2], label="λ R²")
    
    ax_r2.xticks = (1:length(labels), labels)
    axislegend(ax_r2)
    
    return fig
end

"""
    Base.show(io::IO, metrics::Metrics)

Custom display for Metrics objects.
"""
function Base.show(io::IO, metrics::Metrics)
    println(io, "Metrics:")
    println(io, "├─ MSE: $(round(metrics.mse, digits=6))")
    println(io, "├─ MAE: $(round(metrics.mae, digits=6))")
    println(io, "├─ RMSE: $(round(metrics.rmse, digits=6))")
    println(io, "├─ R²: $(round(metrics.r_squared, digits=6))")
    println(io, "├─ Final Loss: $(round(metrics.final_loss, digits=6))")
    println(io, "└─ Training Time: $(round(metrics.training_time, digits=2)) seconds")
end

"""
    Base.show(io::IO, ::MIME"text/plain", metrics::Metrics)

Detailed display for Metrics objects.
"""
function Base.show(io::IO, ::MIME"text/plain", metrics::Metrics)
    println(io, "Metrics:")
    println(io, "├─ MSE: $(round(metrics.mse, digits=6))")
    println(io, "├─ MAE: $(round(metrics.mae, digits=6))")
    println(io, "├─ RMSE: $(round(metrics.rmse, digits=6))")
    println(io, "├─ R²: $(round(metrics.r_squared, digits=6))")
    println(io, "├─ Final Loss: $(round(metrics.final_loss, digits=6))")
    println(io, "├─ Training Time: $(round(metrics.training_time, digits=2)) seconds")
    println(io, "└─ History: $(length(keys(metrics.history))) metrics tracked")
    for (key, values) in metrics.history
        println(io, "   └─ $(key): $(length(values)) points")
    end
end 