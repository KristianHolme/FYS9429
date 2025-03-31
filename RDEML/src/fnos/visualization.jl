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
    xlabel = "Epoch"
    titlesize = 18
    subtitlesize = 12
    ax = Makie.Axis(fig[1, 1]; xlabel, ylabel="Loss", yscale=log10, title, subtitle, titlesize, subtitlesize)
    
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

function plot_test_comparison(;n_t, x, test_data, predicted_states, timesteps=[1, 2, n_t÷2], title="", plot_input=true)
    fig = Figure(size=(600, 180*max(3, length(timesteps))))
    Label(fig[0, 1:2], title, tellwidth=false, fontsize=24)
    
    # Create legend elements for line styles
    # jl_colors = Colors.JULIA_LOGO_COLORS
    colors = [:darkorange1, :blue, :green]
    style_elements = [
        LineElement(color=colors[1], linestyle=:dash, label="Simulation"),
        LineElement(color=colors[2], linestyle=:solid, label="FNO"),
    ]
    style_labels = ["Simulation", "FNO"]
    if plot_input
        push!(style_elements, LineElement(color=colors[3], linestyle=:dash, label="Input"))
        push!(style_labels, "Input")
    end
    
    # legend for line styles
    Legend(fig[1,1:2], 
        style_elements,
        style_labels,
        orientation=:horizontal,
        tellwidth=false
    )
    
    u_axes = []
    λ_axes = []
    # Plot the data
    for (i, timestep) in enumerate(filter(x->x+1<=n_t, timesteps))
        row = i + 1
        Label(fig[row, 0], "t=$(timestep)", fontsize=16, font=:bold, tellwidth=false, tellheight=false)
        ax_u = Makie.Axis(fig[row, 1], xlabel="x", ylabel="u", ylabelrotation=0)
        ax_λ = Makie.Axis(fig[row, 2], xlabel="x", ylabel="λ", ylabelrotation=0)
        push!(u_axes, ax_u)
        push!(λ_axes, ax_λ)
        #plot input data
        if plot_input
            lines!(ax_u, x, test_data[:, 1, timestep], color=colors[3], linestyle=:dash)
            lines!(ax_λ, x, test_data[:, 2, timestep], color=colors[3], linestyle=:dash)
        end
        # plot predicted data
        lines!(ax_u, x, predicted_states[:, 1, timestep+1], color=colors[2])
        lines!(ax_λ, x, predicted_states[:, 2, timestep+1], color=colors[2])
        #plot simulation data
        lines!(ax_u, x, test_data[:, 1, timestep+1], color=colors[1], linestyle=:dash) #+1 because index 1 is time 0
        lines!(ax_λ, x, test_data[:, 2, timestep+1], color=colors[1], linestyle=:dash)
    end
    linkaxes!(u_axes...)
    linkaxes!(λ_axes...)
    colsize!(fig.layout, 0, Relative(1/10))
    
    
    return fig
end

function plot_parameter_analysis(df, param_name::String; 
    title="Parameter Analysis",
    xlabel="Parameter Value",
    ylabel="Final Test Loss",
    window_fraction=0.003,
    save_plots=false,
    experiment_name="parameter_analysis",
    fontsize=18,
    extra_save_dir="")
    
    param_list = skipmissing(sort(unique(df[!, Symbol(param_name)]))) |> collect
    
    fig = Figure(size=(750, 450))
    
    # First plot - Loss history
    ax1 = Axis(fig[1, 1], xlabel="Epoch", ylabel="Loss (moving average)",
        yscale=log10; xlabelsize=fontsize, ylabelsize=fontsize)
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
    axislegend(ax1, ax1_lines, ["Train Loss", "Test Loss"], position=:rt; fontsize)
    
    # Second plot - Individual end loss barplot
    ax2 = Axis(fig[1, 2]; 
        xlabel, 
        ylabel, 
        yscale=log10,
        xticks = (1:length(param_list), string.(param_list)),
        xlabelsize=fontsize, ylabelsize=fontsize)
    
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
    axislegend(ax2, [mean_element], ["Mean Loss"], position=:ct; fontsize)
    
    if save_plots
        dir = plotsdir("fno", experiment_name)
        if !isdir(dir)
            mkdir(dir)
        end
        save(joinpath(dir, "$(param_name)_analysis.svg"), fig)
        save(joinpath(dir, "$(param_name)_analysis.png"), fig)
        if !isempty(extra_save_dir)
            save(joinpath(extra_save_dir, "$(param_name)_analysis.svg"), fig)
            save(joinpath(extra_save_dir, "$(param_name)_analysis.png"), fig)
        end
    end
    
    return fig
end

function plot_training_time_analysis(df, param_name::String; 
    title="Training Time Analysis",
    xlabel="Parameter Value",
    ylabel="Training Time (seconds)",
    save_plots=false,
    experiment_name="parameter_analysis",
    extra_save_dir="",
    fontsize=18)
    
    param_list = sort(unique(df[!, Symbol(param_name)])) |> skipmissing |> collect
    
    fig = Figure(size=(375, 450))
    
    # First plot - Training time vs parameter value
    ax1 = Axis(fig[1, 1]; 
        xlabel, 
        ylabel,
        xticks = (1:length(param_list), string.(param_list)),
        xlabelsize=fontsize,
        ylabelsize=fontsize,
        )
    
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
    
    
    for (i, param_value) in enumerate(param_list)
        group_times = training_times[group .== i]
        group_runs = length(unique(runs[group .== i]))
        mean_time = sum(group_times) / group_runs
        errorbars!(ax1, [i], [mean_time], [0.38f0], color=:black, linewidth=2, direction=:x)
    end
    mean_element = LineElement(color=:black, linewidth=2)
    axislegend(ax1, [mean_element], ["Mean Total Time"], position=:ct; fontsize)
    
    
    if save_plots
        dir = plotsdir("fno", experiment_name)
        if !isdir(dir)
            mkdir(dir)
        end
        basename = "$(param_name)_training_time_analysis"
        save(joinpath(dir, "$basename.svg"), fig)
        save(joinpath(dir, "$basename.png"), fig)
        if !isempty(extra_save_dir)
            save(joinpath(extra_save_dir, "$basename.svg"), fig)
            save(joinpath(extra_save_dir, "$basename.png"), fig)
        end
    end
    
    return fig
end

function plot_losses_final_eval(df;
    folder="final_runs_2",
    window_fraction=0.003,
    save_plot=false,
    fontsize=18,
    extra_save_dir="")
    fig = Figure(size=(600, 400))
    
    # First plot - Loss history
    ax1 = Axis(fig[1, 1], xlabel="Epoch", ylabel="Loss (moving average)",
        yscale=log10; xlabelsize=fontsize, ylabelsize=fontsize)
    
    colors = Makie.Colors.distinguishable_colors(size(df, 1))
    for (i, row) in enumerate(eachrow(df))
        config = row.full_config
        losses = config.history.losses
        total_epochs = sum(config.history.epochs)
        training_progress = collect(1:length(losses)) ./ length(losses)
        epochs = training_progress .* total_epochs
        
        # Plot line
        line = lines!(ax1, epochs, moving_average(losses, Int(floor(length(losses) * window_fraction))),
        color=colors[i])
        # plot test loss
        test_losses = config.history.test_losses
        lines!(ax1, 1:total_epochs, test_losses, linestyle=:dash,
        color=colors[i])
    end
    ax1_lines = [LineElement(color=:black, linestyle=:solid), LineElement(color=:black, linestyle=:dash)]
    ylims!(ax1, (nothing, 10^(-2.5)))
    axislegend(ax1, ax1_lines, ["Train Loss", "Test Loss"], position=:rt; fontsize)

    if save_plot
        dir = plotsdir("fno", folder)
        bname = "final_losses"
        isdir(dir) || mkdir(dir)
        save(joinpath(dir, "$bname.svg"), fig)
        save(joinpath(dir, "$bname.png"), fig)
        if !isempty(extra_save_dir)
            isdir(extra_save_dir) || mkdir(extra_save_dir)
            save(joinpath(extra_save_dir, "$bname.svg"), fig)
            save(joinpath(extra_save_dir, "$bname.png"), fig)
        end
    end

    return fig
end
