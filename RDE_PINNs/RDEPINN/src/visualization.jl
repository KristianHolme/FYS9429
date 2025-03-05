"""
    plot_results(training_result::TrainingResult, simulation_data=nothing; 
                 size=(1200, 1200), colormap=:viridis, λ_colorrange=(0, 1.1))

Plot the results of a training run, optionally comparing with simulation data.
"""
function plot_results(training_result::TrainingResult, simulation_data=nothing; 
                     size=(1200, 1200), colormap=:viridis, λ_colorrange=(0, 1.1))
    
    # Extract data from training result
    ts = training_result.ts
    xs = training_result.xs
    us = training_result.us
    λs = training_result.λs
    
    # Determine color ranges
    u_min, u_max = minimum(minimum.(us)), maximum(maximum.(us))
    
    # Create figure
    fig = Figure(size=size)
    
    # Plot heatmaps for neural network predictions
    ax_u = Axis(fig[1, 1], title="NN u(t, x)", xlabel="t", ylabel="x")
    hm_u = heatmap!(ax_u, ts, xs, stack(us)', colorrange=(u_min, u_max), colormap=colormap)
    Colorbar(fig[1, 3], hm_u)
    
    ax_λ = Axis(fig[1, 2], title="NN λ(t, x)", xlabel="t", ylabel="x")
    hm_λ = heatmap!(ax_λ, ts, xs, stack(λs)', colorrange=λ_colorrange, colormap=colormap)
    Colorbar(fig[1, 4], hm_λ)
    
    # Plot line plots at different time points
    N = length(ts)
    time_indices = [1, Int(round(N/2)), N]
    time_labels = ["t=$(round(ts[i], digits=2))" for i in time_indices]
    
    ax_u_t = Axis(fig[2, 1], title="u(t, x) at different times", xlabel="x", ylabel="u")
    for (i, idx) in enumerate(time_indices)
        lines!(ax_u_t, xs, us[idx], label=time_labels[i], linewidth=2)
    end
    axislegend(ax_u_t)
    
    ax_λ_t = Axis(fig[2, 2], title="λ(t, x) at different times", xlabel="x", ylabel="λ")
    for (i, idx) in enumerate(time_indices)
        lines!(ax_λ_t, xs, λs[idx], label=time_labels[i], linewidth=2)
    end
    ylims!(ax_λ_t, λ_colorrange...)
    axislegend(ax_λ_t)
    
    # Add loss information
    loss_text = "Final loss: $(round(training_result.final_loss, digits=6))\n" *
                "PDE losses: $(round.(training_result.pde_losses, digits=6))\n" *
                "BC losses: $(round.(training_result.bc_losses, digits=6))"
    
    Label(fig[3, :], loss_text, tellwidth=false)
    
    # If simulation data is provided, add comparison plots
    if simulation_data !== nothing
        ts_sim, xs_sim, us_sim, λs_sim = simulation_data
        
        # Update color ranges to include simulation data
        u_min = min(u_min, minimum(minimum.(us_sim)))
        u_max = max(u_max, maximum(maximum.(us_sim)))
        
        # Update heatmap color ranges
        hm_u.colorrange = (u_min, u_max)
        
        # Add simulation heatmaps
        ax_u_sim = Axis(fig[4, 1], title="Sim u(t, x)", xlabel="t", ylabel="x")
        hm_u_sim = heatmap!(ax_u_sim, ts_sim, xs_sim, stack(us_sim)', 
                           colorrange=(u_min, u_max), colormap=colormap)
        Colorbar(fig[4, 3], hm_u_sim)
        
        ax_λ_sim = Axis(fig[4, 2], title="Sim λ(t, x)", xlabel="t", ylabel="x")
        hm_λ_sim = heatmap!(ax_λ_sim, ts_sim, xs_sim, stack(λs_sim)', 
                           colorrange=λ_colorrange, colormap=colormap)
        Colorbar(fig[4, 4], hm_λ_sim)
        
        # Add simulation line plots
        N_sim = length(ts_sim)
        sim_time_indices = [1, Int(round(N_sim/2)), N_sim]
        sim_time_labels = ["t=$(round(ts_sim[i], digits=2))" for i in sim_time_indices]
        
        ax_u_sim_t = Axis(fig[5, 1], title="Sim u(t, x) at different times", xlabel="x", ylabel="u")
        linkaxes!(ax_u_sim_t, ax_u_t)
        for (i, idx) in enumerate(sim_time_indices)
            lines!(ax_u_sim_t, xs_sim, us_sim[idx], label=sim_time_labels[i], linewidth=2)
        end
        axislegend(ax_u_sim_t)
        
        ax_λ_sim_t = Axis(fig[5, 2], title="Sim λ(t, x) at different times", xlabel="x", ylabel="λ")
        linkaxes!(ax_λ_sim_t, ax_λ_t)
        for (i, idx) in enumerate(sim_time_indices)
            lines!(ax_λ_sim_t, xs_sim, λs_sim[idx], label=sim_time_labels[i], linewidth=2)
        end
        ylims!(ax_λ_sim_t, λ_colorrange...)
        axislegend(ax_λ_sim_t)
        
        # Add error plots
        if length(ts) == length(ts_sim) && length(xs) == length(xs_sim)
            u_errors = [abs.(us[i] - us_sim[i]) for i in 1:length(ts)]
            λ_errors = [abs.(λs[i] - λs_sim[i]) for i in 1:length(ts)]
            
            max_u_error = maximum(maximum.(u_errors))
            max_λ_error = maximum(maximum.(λ_errors))
            
            ax_u_error = Axis(fig[6, 1], title="u(t, x) absolute error", xlabel="t", ylabel="x")
            hm_u_error = heatmap!(ax_u_error, ts, xs, stack(u_errors)', 
                                 colorrange=(0, max_u_error), colormap=:thermal)
            Colorbar(fig[6, 3], hm_u_error)
            
            ax_λ_error = Axis(fig[6, 2], title="λ(t, x) absolute error", xlabel="t", ylabel="x")
            hm_λ_error = heatmap!(ax_λ_error, ts, xs, stack(λ_errors)', 
                                 colorrange=(0, max_λ_error), colormap=:thermal)
            Colorbar(fig[6, 4], hm_λ_error)
        end
    end
    
    return fig
end

"""
    plot_comparison(results::Vector{TrainingResult}, labels::Vector{String}; 
                   size=(1200, 800), time_indices=[1, -1])

Compare multiple training results.
"""
function plot_comparison(results::Vector{TrainingResult}, labels::Vector{String}; 
                        size=(1200, 800), time_indices=[1, -1])
    
    n_results = length(results)
    if n_results != length(labels)
        error("Number of results ($(n_results)) must match number of labels ($(length(labels)))")
    end
    
    # Create figure
    fig = Figure(size=size)
    
    # Plot u and λ for each result at specified time indices
    for (i, time_idx) in enumerate(time_indices)
        time_label = time_idx == 1 ? "Initial" : (time_idx == -1 ? "Final" : "t=$(time_idx)")
        
        # Plot u
        ax_u = Axis(fig[i, 1], title="u(x) - $(time_label)", xlabel="x", ylabel="u")
        
        for (j, result) in enumerate(results)
            actual_idx = time_idx == -1 ? length(result.ts) : time_idx
            lines!(ax_u, result.xs, result.us[actual_idx], label=labels[j], linewidth=2)
        end
        
        axislegend(ax_u)
        
        # Plot λ
        ax_λ = Axis(fig[i, 2], title="λ(x) - $(time_label)", xlabel="x", ylabel="λ")
        
        for (j, result) in enumerate(results)
            actual_idx = time_idx == -1 ? length(result.ts) : time_idx
            lines!(ax_λ, result.xs, result.λs[actual_idx], label=labels[j], linewidth=2)
        end
        
        ylims!(ax_λ, 0, 1.1)
        axislegend(ax_λ)
    end
    
    # Plot losses
    ax_losses = Axis(fig[length(time_indices)+1, :], title="Losses", xlabel="Model", ylabel="Loss (log scale)")
    
    losses = [result.final_loss for result in results]
    barplot!(ax_losses, 1:n_results, losses, labels=labels)
    ax_losses.yscale = log10
    
    return fig
end

"""
    plot_spacetime_slices(training_result::TrainingResult, simulation_data=nothing; 
                         x_slices=[0.25, 0.5, 0.75], size=(1200, 800))

Plot time evolution at specific spatial points.
"""
function plot_spacetime_slices(training_result::TrainingResult, simulation_data=nothing; 
                              x_slices=[0.25, 0.5, 0.75], size=(1200, 800))
    
    # Extract data from training result
    ts = training_result.ts
    xs = training_result.xs
    us = training_result.us
    λs = training_result.λs
    
    # Create figure
    fig = Figure(size=size)
    
    # Find indices of x_slices
    x_indices = [findmin(abs.(xs .- x_slice))[2] for x_slice in x_slices]
    x_values = [xs[idx] for idx in x_indices]
    
    # Plot u over time at each x slice
    ax_u = Axis(fig[1, 1], title="u(t) at fixed x", xlabel="t", ylabel="u")
    
    for (i, idx) in enumerate(x_indices)
        u_at_x = [us[t_idx][idx] for t_idx in 1:length(ts)]
        lines!(ax_u, ts, u_at_x, label="x=$(round(x_values[i], digits=2))", linewidth=2)
    end
    
    axislegend(ax_u)
    
    # Plot λ over time at each x slice
    ax_λ = Axis(fig[1, 2], title="λ(t) at fixed x", xlabel="t", ylabel="λ")
    
    for (i, idx) in enumerate(x_indices)
        λ_at_x = [λs[t_idx][idx] for t_idx in 1:length(ts)]
        lines!(ax_λ, ts, λ_at_x, label="x=$(round(x_values[i], digits=2))", linewidth=2)
    end
    
    ylims!(ax_λ, 0, 1.1)
    axislegend(ax_λ)
    
    # If simulation data is provided, add comparison
    if simulation_data !== nothing
        ts_sim, xs_sim, us_sim, λs_sim = simulation_data
        
        # Find indices of x_slices in simulation data
        x_sim_indices = [findmin(abs.(xs_sim .- x_slice))[2] for x_slice in x_slices]
        x_sim_values = [xs_sim[idx] for idx in x_sim_indices]
        
        # Plot simulation u over time
        ax_u_sim = Axis(fig[2, 1], title="Sim u(t) at fixed x", xlabel="t", ylabel="u")
        linkaxes!(ax_u_sim, ax_u)
        
        for (i, idx) in enumerate(x_sim_indices)
            u_at_x = [us_sim[t_idx][idx] for t_idx in 1:length(ts_sim)]
            lines!(ax_u_sim, ts_sim, u_at_x, label="x=$(round(x_sim_values[i], digits=2))", linewidth=2)
        end
        
        axislegend(ax_u_sim)
        
        # Plot simulation λ over time
        ax_λ_sim = Axis(fig[2, 2], title="Sim λ(t) at fixed x", xlabel="t", ylabel="λ")
        linkaxes!(ax_λ_sim, ax_λ)
        
        for (i, idx) in enumerate(x_sim_indices)
            λ_at_x = [λs_sim[t_idx][idx] for t_idx in 1:length(ts_sim)]
            lines!(ax_λ_sim, ts_sim, λ_at_x, label="x=$(round(x_sim_values[i], digits=2))", linewidth=2)
        end
        
        ylims!(ax_λ_sim, 0, 1.1)
        axislegend(ax_λ_sim)
        
        # Add direct comparison plots if time grids are compatible
        if length(ts) == length(ts_sim)
            for (i, (idx, sim_idx)) in enumerate(zip(x_indices, x_sim_indices))
                ax_comp = Axis(fig[3, i], title="Comparison at x=$(round(x_values[i], digits=2))", 
                              xlabel="t", ylabel="Value")
                
                u_nn = [us[t_idx][idx] for t_idx in 1:length(ts)]
                u_sim = [us_sim[t_idx][sim_idx] for t_idx in 1:length(ts)]
                
                λ_nn = [λs[t_idx][idx] for t_idx in 1:length(ts)]
                λ_sim = [λs_sim[t_idx][sim_idx] for t_idx in 1:length(ts)]
                
                lines!(ax_comp, ts, u_nn, label="NN u", linewidth=2)
                lines!(ax_comp, ts, u_sim, label="Sim u", linewidth=2, linestyle=:dash)
                lines!(ax_comp, ts, λ_nn, label="NN λ", linewidth=2)
                lines!(ax_comp, ts, λ_sim, label="Sim λ", linewidth=2, linestyle=:dash)
                
                axislegend(ax_comp)
            end
        end
    end
    
    return fig
end

"""
    animate_solution(training_result::TrainingResult, simulation_data=nothing; 
                    filename="rde_solution.mp4", framerate=30)

Create an animation of the solution.
"""
function animate_solution(training_result::TrainingResult, simulation_data=nothing; 
                         filename="rde_solution.mp4", framerate=30)
    
    # Extract data
    ts = training_result.ts
    xs = training_result.xs
    us = training_result.us
    λs = training_result.λs
    
    # Determine color ranges
    u_min, u_max = minimum(minimum.(us)), maximum(maximum.(us))
    
    if simulation_data !== nothing
        _, _, us_sim, λs_sim = simulation_data
        u_min = min(u_min, minimum(minimum.(us_sim)))
        u_max = max(u_max, maximum(maximum.(us_sim)))
    end
    
    # Create figure for animation
    fig = Figure(size=(1200, 600))
    
    # Create axes
    ax_u = Axis(fig[1, 1], title="u(t, x)", xlabel="x", ylabel="u")
    ax_λ = Axis(fig[1, 2], title="λ(t, x)", xlabel="x", ylabel="λ")
    
    # Set axis limits
    ylims!(ax_u, u_min, u_max)
    ylims!(ax_λ, 0, 1.1)
    
    # Create initial plots
    u_line = lines!(ax_u, xs, us[1], linewidth=2, label="NN")
    λ_line = lines!(ax_λ, xs, λs[1], linewidth=2, label="NN")
    
    # Add simulation data if provided
    if simulation_data !== nothing
        _, xs_sim, us_sim, λs_sim = simulation_data
        u_sim_line = lines!(ax_u, xs_sim, us_sim[1], linewidth=2, linestyle=:dash, label="Sim")
        λ_sim_line = lines!(ax_λ, xs_sim, λs_sim[1], linewidth=2, linestyle=:dash, label="Sim")
        axislegend(ax_u)
        axislegend(ax_λ)
    end
    
    # Add time label
    time_label = Label(fig[0, :], "t = $(ts[1])")
    
    # Create animation
    record(fig, filename, 1:length(ts); framerate=framerate) do frame
        # Update lines
        u_line[2] = us[frame]
        λ_line[2] = λs[frame]
        
        # Update simulation lines if provided
        if simulation_data !== nothing
            # Handle potential different time grids
            sim_frame = min(frame, length(us_sim))
            u_sim_line[2] = us_sim[sim_frame]
            λ_sim_line[2] = λs_sim[sim_frame]
        end
        
        # Update time label
        time_label.text = "t = $(round(ts[frame], digits=3))"
    end
    
    return filename
end

"""
    plot_solution(ts, xs, us, λs; title="Rotating Detonation Engine Solution")

Plot the solution of a Rotating Detonation Engine simulation.
"""
function plot_solution(ts, xs, us, λs; title="Rotating Detonation Engine Solution")
    fig = Figure(size=(1200, 600))
    
    # Plot u
    ax1 = Axis(fig[1, 1], title="u(t,x)", xlabel="x", ylabel="t")
    hm1 = heatmap!(ax1, xs, ts, us, colormap=:viridis)
    Colorbar(fig[1, 2], hm1)
    
    # Plot λ
    ax2 = Axis(fig[1, 3], title="λ(t,x)", xlabel="x", ylabel="t")
    hm2 = heatmap!(ax2, xs, ts, λs, colormap=:viridis)
    Colorbar(fig[1, 4], hm2)
    
    Label(fig[0, 1:4], title, fontsize=20)
    
    return fig
end

"""
    plot_spacetime_slice(ts, xs, us, λs, t_idx; title="Rotating Detonation Engine Spacetime Slice")

Plot a spacetime slice of a Rotating Detonation Engine simulation at a specific time index.
"""
function plot_spacetime_slice(ts, xs, us, λs, t_idx; title="Rotating Detonation Engine Spacetime Slice")
    fig = Figure(size=(1000, 400))
    
    t_val = ts[t_idx]
    
    # Plot u
    ax1 = Axis(fig[1, 1], title="u(t=$t_val,x)", xlabel="x", ylabel="u")
    lines!(ax1, xs, us[t_idx], linewidth=2, color=:blue)
    
    # Plot λ
    ax2 = Axis(fig[1, 2], title="λ(t=$t_val,x)", xlabel="x", ylabel="λ")
    lines!(ax2, xs, λs[t_idx], linewidth=2, color=:red)
    
    Label(fig[0, 1:2], title, fontsize=20)
    
    return fig
end

"""
    plot_comparison(ts, xs, us, λs, us_sim, λs_sim; title="RDE PINN vs Simulation Comparison")

Compare PINN solution with simulation for Rotating Detonation Engine.
"""
function plot_comparison(ts, xs, us, λs, us_sim, λs_sim; title="RDE PINN vs Simulation Comparison")
    fig = Figure(size=(1200, 800))
    
    # Plot u from PINN
    ax1 = Axis(fig[1, 1], title="PINN u(t,x)", xlabel="x", ylabel="t")
    hm1 = heatmap!(ax1, xs, ts, us, colormap=:viridis)
    Colorbar(fig[1, 2], hm1)
    
    # Plot u from simulation
    ax2 = Axis(fig[1, 3], title="Simulation u(t,x)", xlabel="x", ylabel="t")
    hm2 = heatmap!(ax2, xs, ts, us_sim, colormap=:viridis)
    Colorbar(fig[1, 4], hm2)
    
    # Plot λ from PINN
    ax3 = Axis(fig[2, 1], title="PINN λ(t,x)", xlabel="x", ylabel="t")
    hm3 = heatmap!(ax3, xs, ts, λs, colormap=:viridis)
    Colorbar(fig[2, 2], hm3)
    
    # Plot λ from simulation
    ax4 = Axis(fig[2, 3], title="Simulation λ(t,x)", xlabel="x", ylabel="t")
    hm4 = heatmap!(ax4, xs, ts, λs_sim, colormap=:viridis)
    Colorbar(fig[2, 4], hm4)
    
    Label(fig[0, 1:4], title, fontsize=20)
    
    return fig
end

"""
    plot_error(ts, xs, us, λs, us_sim, λs_sim; title="RDE PINN Error Analysis")

Plot the error between PINN solution and simulation for Rotating Detonation Engine.
"""
function plot_error(ts, xs, us, λs, us_sim, λs_sim; title="RDE PINN Error Analysis")
    # Calculate errors
    u_error = abs.(us - us_sim)
    λ_error = abs.(λs - λs_sim)
    
    fig = Figure(size=(1000, 500))
    
    # Plot u error
    ax1 = Axis(fig[1, 1], title="u(t,x) Error", xlabel="x", ylabel="t")
    hm1 = heatmap!(ax1, xs, ts, u_error, colormap=:thermal)
    Colorbar(fig[1, 2], hm1)
    
    # Plot λ error
    ax2 = Axis(fig[1, 3], title="λ(t,x) Error", xlabel="x", ylabel="t")
    hm2 = heatmap!(ax2, xs, ts, λ_error, colormap=:thermal)
    Colorbar(fig[1, 4], hm2)
    
    Label(fig[0, 1:4], title, fontsize=20)
    
    return fig
end

"""
    create_animation(ts, xs, us, λs; filename="rde_animation.mp4", framerate=15)

Create an animation of the Rotating Detonation Engine solution over time.
"""
function create_animation(ts, xs, us, λs; filename="rde_animation.mp4", framerate=15)
    fig = Figure(size=(1000, 400))
    
    # Plot u
    ax1 = Axis(fig[1, 1], title="u(t,x)", xlabel="x", ylabel="u")
    lines_u = lines!(ax1, xs, us[1], linewidth=2, color=:blue)
    
    # Plot λ
    ax2 = Axis(fig[1, 2], title="λ(t,x)", xlabel="x", ylabel="λ")
    lines_λ = lines!(ax2, xs, λs[1], linewidth=2, color=:red)
    
    # Create time label
    time_label = Label(fig[0, 1:2], "t = $(ts[1])", fontsize=20)
    
    # Create animation
    record(fig, filename, 1:length(ts); framerate=framerate) do i
        lines_u[1] = xs
        lines_u[2] = us[i]
        
        lines_λ[1] = xs
        lines_λ[2] = λs[i]
        
        time_label.text = "t = $(round(ts[i], digits=3))"
    end
    
    return filename
end

"""
    plot_experiment_comparison(experiments; metric="final_loss", title="RDE Experiment Comparison")

Compare multiple Rotating Detonation Engine experiments based on a specific metric.
"""
function plot_experiment_comparison(experiments; metric="final_loss", title="RDE Experiment Comparison")
    fig = Figure(size=(800, 500))
    
    ax = Axis(fig[1, 1], title=title, xlabel="Experiment", ylabel=metric)
    
    # Extract experiment names and metric values
    names = [exp.name for exp in experiments]
    values = [getfield(exp.metrics, Symbol(metric)) for exp in experiments]
    
    # Create bar plot
    barplot!(ax, 1:length(names), values)
    ax.xticks = (1:length(names), names)
    ax.xticklabelrotation = π/4
    
    return fig
end

"""
    plot_metrics_over_time(experiment; metrics=["loss"], title="RDE Training Metrics")

Plot metrics over time for a Rotating Detonation Engine experiment.
"""
function plot_metrics_over_time(experiment; metrics=["loss"], title="RDE Training Metrics")
    fig = Figure(size=(800, 500))
    
    ax = Axis(fig[1, 1], title=title, xlabel="Iteration", ylabel="Value")
    
    # Plot each metric
    for metric in metrics
        if haskey(experiment.metrics.history, Symbol(metric))
            values = experiment.metrics.history[Symbol(metric)]
            iterations = 1:length(values)
            lines!(ax, iterations, values, label=metric)
        end
    end
    
    axislegend(ax)
    
    return fig
end 