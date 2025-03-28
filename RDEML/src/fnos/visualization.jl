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

function plot_parameter_analysis(df, param_name::String; 
    title="Parameter Analysis",
    xlabel="Parameter Value",
    ylabel="Final Test Loss",
    window_fraction=0.1,
    save_plots=false,
    experiment_name="parameter_analysis")
    
    param_list = skipmissing(sort(unique(df[!, Symbol(param_name)]))) |> collect
    
    fig = Figure(size=(1000, 600))
    
    # First plot - Loss history
    ax1 = Axis(fig[1, 1], xlabel="Epoch", ylabel="Loss (moving average)", yscale=log10)
    for (i, row) in enumerate(eachrow(df))
        config = row.full_config
        losses = config.history.losses
        total_epochs = sum(config.history.epochs)
        n_batches = length(losses) ÷ total_epochs
        training_progress = collect(1:length(losses)) ./ length(losses)
        epochs = training_progress .* total_epochs
        
        # Get color for this parameter value
        param_value = row[Symbol(param_name)]
        color_idx = indexin(param_value, param_list)[1]
        color = Makie.wong_colors()[color_idx]
        
        # Plot line
        line = lines!(ax1, epochs, moving_average(losses, Int(floor(length(losses) * window_fraction))), color=color)
        # plot test loss
        test_losses = config.history.test_losses
        lines!(ax1, 1:total_epochs, test_losses, color=color, linestyle=:dash)
    end
    ax1_lines = [LineElement(color=:black, linestyle=:solid), LineElement(color=:black, linestyle=:dash)]
    axislegend(ax1, ax1_lines, ["Train Loss", "Test Loss"], position=:rt)
    
    # Second plot - Individual end loss barplot
    ax2 = Axis(fig[1, 2], 
        xlabel=xlabel, 
        ylabel=ylabel, 
        yscale=log10,
        xticks = (1:length(param_list), string.(param_list)))
    
    # Collect final losses and parameter values
    final_train_losses = df.final_train_loss |> skipmissing |> collect
    final_test_losses = df.final_test_loss |> skipmissing |> collect
    param_values = df[!, Symbol(param_name)] |> skipmissing |> collect
    group = [indexin(value, param_list)[1] for value in param_values]
    run_id = df.run |> skipmissing |> collect
    
    wcolors = Makie.wong_colors()[1:length(param_list)]
    colors = wcolors[group]
    barplot!(ax2, group, final_test_losses, 
        dodge=run_id, color = colors)
    
    # Calculate mean loss for each parameter group
    for (i, param_value) in enumerate(param_list)
        group_losses = final_test_losses[group .== i]
        mean_loss = mean(group_losses)
        errorbars!(ax2, [i], [mean_loss], [0.38f0], color=:black, linewidth=2, direction=:x)
    end
    mean_element = LineElement(color=:black, linewidth=2)
    axislegend(ax2, [mean_element], ["Mean Loss"], position=:lt)
    
    if save_plots
        dir = plotsdir("fno", experiment_name)
        if !isdir(dir)
            mkdir(dir)
        end
        save(joinpath(dir, "$(param_name)_analysis.svg"), fig)
        save(joinpath(dir, "$(param_name)_analysis.png"), fig)
    end
    
    return fig
end

function plot_training_time_analysis(df, param_name::String; 
    title="Training Time Analysis",
    xlabel="Parameter Value",
    ylabel="Training Time (seconds)",
    save_plots=false,
    experiment_name="parameter_analysis")
    
    param_list = sort(unique(df[!, Symbol(param_name)])) |> skipmissing |> collect
    
    fig = Figure(size=(500, 600))
    
    # First plot - Training time vs parameter value
    ax1 = Axis(fig[1, 1], 
        xlabel=xlabel, 
        ylabel=ylabel,
        xticks = (1:length(param_list), string.(param_list)))
    
    # Collect training times and parameter values
    param_values = df[!, Symbol(param_name)] |> skipmissing |> collect
    run_id = df.run |> skipmissing |> collect
    
    # Get training times for each learning rate and epoch combination
    training_times = Float32[]
    training_periods = Int[]
    runs = Int[]
    group = Int[]

    for (i, row) in enumerate(eachrow(df))
        config = row.full_config
        times = config.history.training_time
        push!(training_times, times...)
        push!(training_periods, collect(1:length(times))...)
        push!(runs, fill(run_id[i], length(times))...)
        push!(group, fill(indexin(param_values[i], param_list)[1], length(times))...)
    end
    @show typeof(training_times), typeof(training_periods), typeof(runs), typeof(group)
    @show size(training_times), size(training_periods), size(runs), size(group)
    
    # Create stacked barplot
    wcolors = Makie.wong_colors()[1:length(param_list)]
    colors = wcolors[group]
    

    barplot!(ax1, group, training_times,
        stack=training_periods,
        dodge=runs,
        color=colors)
    
    # Calculate mean total training time for each parameter group
    total_times = [sum(times) for times in training_times]
    for (i, param_value) in enumerate(param_list)
        group_times = total_times[group .== i]
        mean_time = mean(group_times)
        errorbars!(ax1, [i], [mean_time], [0.38f0], color=:black, linewidth=2, direction=:x)
    end
    mean_element = LineElement(color=:black, linewidth=2)
    axislegend(ax1, [mean_element], ["Mean Total Time"], position=:lt)
    
    
    if save_plots
        dir = plotsdir("fno", experiment_name)
        if !isdir(dir)
            mkdir(dir)
        end
        save(joinpath(dir, "$(param_name)_training_time_analysis.svg"), fig)
        save(joinpath(dir, "$(param_name)_training_time_analysis.png"), fig)
    end
    
    return fig
end