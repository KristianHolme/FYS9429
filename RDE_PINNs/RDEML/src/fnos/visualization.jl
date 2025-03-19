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

function plot_losses(losses, title="Losses")
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

function plot_losses(fno_config)
    title = "Losses for $(fno_config.chs), $(fno_config.modes), $(fno_config.activation)"
    plot_losses(fno_config.history.losses, title)
end

function plot_test_comparison(;n_t, x, test_data, predicted_states, times=[1, 2, n_t÷2], title="")
    fig = Figure(size=(1000, 200*length(times)))
    Label(fig[0, 1:2], title, tellwidth=false, fontsize=24)
    
    # Create legend elements for line styles
    style_elements = [
        LineElement(color=:black, linestyle=:dash, label="Simulation"),
        LineElement(color=:black, linestyle=:solid, label="FNO")
    ]
    
    
    colors = Makie.wong_colors()[1:2]
    
    # Plot the data
    for (i, time_step) in enumerate(times)
        Label(fig[i*2, :], "t=$(times[i])", fontsize=16, font=:bold)
        ax_u = Makie.Axis(fig[i*2+1, 1], xlabel="x", ylabel="u")
        ax_λ = Makie.Axis(fig[i*2+1, 2], xlabel="x", ylabel="λ")
        lines!(ax_u, x, test_data[:, 1, time_step+1], color=colors[1], linestyle=:dash) #+1 because index 1 is time 0
        lines!(ax_u, x, predicted_states[:, 1, time_step+1], color=colors[1])
        lines!(ax_λ, x, test_data[:, 2, time_step+1], color=colors[2], linestyle=:dash)
        lines!(ax_λ, x, predicted_states[:, 2, time_step+1], color=colors[2])
    end
    # legend for line styles
    Legend(fig[1,1:2], 
        style_elements,
        ["Simulation", "FNO"],
        orientation=:horizontal,
        tellwidth=false
    )
    
    
    return fig
end