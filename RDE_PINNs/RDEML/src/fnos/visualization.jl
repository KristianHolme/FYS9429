function visualize_data(run_data, policies, envs, reset_strategies, 
    n_runs_per_reset_strategy; save_plots=false, display_plots=true)
    n_runs = length(reset_strategies)*n_runs_per_reset_strategy
    for (i, rdata) in enumerate(run_data)
        policy_idx = div(i-1, n_runs) + 1
        total_run_idx = mod(i-1, n_runs) + 1
        reset_strategy_idx = div(total_run_idx-1, n_runs_per_reset_strategy) + 1
        reset_strategy_run_idx = mod(total_run_idx-1, n_runs_per_reset_strategy) + 1
        name = "$(typeof(policies[policy_idx])) ($(policy_idx)), $(reset_strategies[reset_strategy_idx]) ( $(reset_strategy_idx) ), run $reset_strategy_run_idx"
        fig = plot_shifted_history(rdata, envs[policy_idx].prob.x, use_rewards=false,
            title=name)
        if save_plots
            safesave(joinpath(plotsdir(), "data_collection_viz", name*".png"), fig)
        end
        display_plots && display(fig)
    end
end

function plot_losses(fno_config, losses)
    fig = Figure()
    ax = Makie.Axis(fig[1, 1], xlabel="Epoch", ylabel="Loss", xscale=log10, yscale=log10)

    # Calculate smoothed line and confidence bands
    window_size = 50
    smoothed = [mean(losses[max(1,i-window_size):i]) for i in 1:length(losses)]
    upper = [quantile(losses[max(1,i-window_size):i], 0.95) for i in 1:length(losses)]
    lower = [quantile(losses[max(1,i-window_size):i], 0.05) for i in 1:length(losses)]

    # Plot bands and smoothed line
    band!(ax, 1:length(losses), lower, upper, color=(:blue, 0.2))
    lines!(ax, smoothed, color=:blue, linewidth=2)
    return fig
end

function plot_test_comparison(;n_t, test_data, output_data, times=[1, 2, n_t÷2])
    fig = Figure()
    ax_u = Makie.Axis(fig[1, 1], xlabel="Time", ylabel="u")
    ax_λ = Makie.Axis(fig[2, 1], xlabel="Time", ylabel="λ")
    
    colors = Makie.wong_colors()[1:length(times)]
    
    # Create legend elements for line styles
    style_elements = [
        LineElement(color=:black, linestyle=:dash, label="Simulation"),
        LineElement(color=:black, linestyle=:solid, label="FNO")
    ]
    
    # Create legend elements for timesteps
    time_elements = [LineElement(color=colors[i], label="t=$(times[i])") for i in 1:length(times)]
    
    # Plot the data
    for (i, ind) in enumerate(times)
        lines!(ax_u, test_data[:, 1, ind+1], color=colors[i], linestyle=:dash)
        lines!(ax_u, output_data[:, 1, ind], color=colors[i])
        lines!(ax_λ, test_data[:, 2, ind+1], color=colors[i], linestyle=:dash)
        lines!(ax_λ, output_data[:, 2, ind], color=colors[i])
    end
    
    # Create single multi-group legend
    Legend(fig[1:2,2], 
        [style_elements, time_elements],
        [["Simulation", "FNO"], ["t=$(times[i])" for i in 1:length(times)]],
        ["Style", "Timestep"],
        vertical=true,
        tellheight=false)
    
    return fig
end