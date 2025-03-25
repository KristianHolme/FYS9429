function visualize_data(df, env=make_env(); save_plots=false, display_plots=true)
    for df_row in eachrow(df)
        data_info = wload(df_row.path, DataSetInfo)
        policy = data_info.policy
        reset_strategy = data_info.reset_strategy
        run_idx = data_info.run
        name = "$(policy), $(reset_strategy), run $run_idx"
        fig = plot_shifted_history(data_info.sim_data, env.prob.x, use_rewards=false,
            title=name)
        if save_plots
            safesave(joinpath(plotsdir(), "data_collection_viz", name*".png"), fig)
        end
        display_plots && display(fig)
    end
end


function plot_losses(fno_config::FNOConfig; 
                     saveplot=false, 
                     folder="", 
                     title="Losses",
                     x_axis_mode=:steps,
                     include_test_losses=true,
                     window_size=50,
                     plot_confidence_bands=true,
                     train_color=:blue,
                     test_color=:red,
                     kwargs...)
    
    subtitle = "FNO config: $(fno_config.chs), $(fno_config.modes), $(fno_config.activation)"
    history = fno_config.history
    losses = history.losses
    epochs = history.epochs
    test_losses = history.test_losses
    
    fig = Figure()
    Label(fig[1, 1], title, tellwidth=false, tellheight=true, fontsize=18)
    Label(fig[end+1, 1], subtitle, tellwidth=false, tellheight=true, fontsize=12)
    
    # Determine x-axis label based on mode
    x_label = "Epoch"
    ax = Makie.Axis(fig[end+1, 1], xlabel=x_label, ylabel="Loss", yscale=log10)
    
    # Calculate x values based on mode
    @info "epochs: $(sum(epochs)), length(losses): $(length(losses))"
    x_train = collect(1:length(losses)) ./ length(losses) .* sum(epochs)

    
    # Calculate smoothed line and confidence bands for training losses
    window_size = min(window_size, length(losses))
    smoothed = [mean(losses[max(1,i-window_size):i]) for i in 1:length(losses)]
    @info "lenght losses: $(length(losses))"
    # Plot training loss
    if plot_confidence_bands && length(losses) > window_size
        upper = [quantile(losses[max(1,i-window_size):i], 0.95) for i in 1:length(losses)]
        lower = [quantile(losses[max(1,i-window_size):i], 0.05) for i in 1:length(losses)]
        @info "x_train: $(size(x_train)), upper: $(size(upper)), lower: $(size(lower))"
        band!(ax, x_train, lower, upper, color=(train_color, 0.2))
    end
    
    train_line = lines!(ax, x_train, smoothed, color=train_color, linewidth=2)
    
    # Plot test losses if provided
    if !isnothing(test_losses) && !isempty(test_losses)
        # For test losses, x values are epochs (1-based)
        x_test = collect(1:length(test_losses)) ./ length(test_losses) .* sum(epochs)
        test_line = lines!(ax, x_test, test_losses, color=test_color, linewidth=2, linestyle=:dash)
        
        # Add legend
        axislegend(ax, 
            [train_line, test_line], 
            ["Training Loss", "Test Loss"],
            tellwidth=false,
            orientation=:horizontal,
            position=:rt)
    end
    
    
    if saveplot
        path = plotsdir("fno", folder, savename(fno_config, "svg"))
        if !isdir(plotsdir("fno", folder))
            mkdir(plotsdir("fno", folder))
        end
        save(path, fig)
    end
    
    return fig
end

function plot_test_comparison(;n_t, x, test_data, predicted_states, timesteps=[1, 2, n_t÷2], title="")
    fig = Figure(size=(1000, 200*length(timesteps)))
    Label(fig[0, 1:2], title, tellwidth=false, fontsize=24)
    
    # Create legend elements for line styles
    style_elements = [
        LineElement(color=:black, linestyle=:dash, label="Simulation"),
        LineElement(color=:black, linestyle=:solid, label="FNO")
    ]
    
    
    colors = Makie.wong_colors()[1:2]
    
    # Plot the data
    for (i, timestep) in enumerate(filter(x->x+1<=n_t, timesteps))
        Label(fig[i*2, :], "t=$(timestep)", fontsize=16, font=:bold)
        ax_u = Makie.Axis(fig[i*2+1, 1], xlabel="x", ylabel="u", ylabelrotation=0)
        ax_λ = Makie.Axis(fig[i*2+1, 2], xlabel="x", ylabel="λ", ylabelrotation=0)
        lines!(ax_u, x, test_data[:, 1, timestep+1], color=colors[1], linestyle=:dash) #+1 because index 1 is time 0
        lines!(ax_u, x, predicted_states[:, 1, timestep+1], color=colors[1])
        lines!(ax_λ, x, test_data[:, 2, timestep+1], color=colors[2], linestyle=:dash)
        lines!(ax_λ, x, predicted_states[:, 2, timestep+1], color=colors[2])
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