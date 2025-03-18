using JLD2
using Dates
using DrWatson

"""
    Experiment

Structure to store information about a Rotating Detonation Engine (RDE) PINN experiment.
"""
struct ExperimentSetup
    name::String
    pde_config::PDESystemConfig
    model_config::ModelConfig
    training_config::TrainingConfig
end
struct ExperimentResults
    metrics::Metrics
    predictions::Dict{Symbol, Any}
    minimizers
    timestamp::DateTime
end
function ExperimentResults(d::Dict)
    return ExperimentResults(d["metrics"], d["predictions"], d["minimizers"], d["timestamp"])
end

"""
    Base.show(io::IO, setup::ExperimentSetup)

Custom display for ExperimentSetup objects.
"""
function Base.show(io::IO, setup::ExperimentSetup)
    println(io, "ExperimentSetup: $(setup.name)")
    println(io, "├─ Model: $(length(setup.model_config.hidden_sizes)) layers $(setup.model_config.hidden_sizes)")
    println(io, "├─ PDE: u_scale=$(setup.pde_config.u_scale), tmax=$(setup.pde_config.rde_params.tmax)")
    println(io, "└─ Training: $(length(setup.training_config.iterations)) stages, $(sum(setup.training_config.iterations)) total iterations")
end

"""
    Base.show(io::IO, results::ExperimentResults)

Custom display for ExperimentResults objects.
"""
function Base.show(io::IO, results::ExperimentResults)
    println(io, "ExperimentResults:")
    println(io, "├─ MSE: $(round(results.metrics.mse, digits=6))")
    println(io, "├─ RMSE: $(round(results.metrics.rmse, digits=6))")
    println(io, "├─ Final Loss: $(round(results.metrics.final_loss, digits=6))")
    println(io, "├─ Training Time: $(round(results.metrics.training_time, digits=2)) seconds")
    println(io, "└─ Timestamp: $(results.timestamp)")
end

"""
    Base.show(io::IO, ::MIME"text/plain", setup::ExperimentSetup)

Detailed display for ExperimentSetup objects.
"""
function Base.show(io::IO, ::MIME"text/plain", setup::ExperimentSetup)
    println(io, "ExperimentSetup: $(setup.name)")
    println(io, "├─ Model Configuration:")
    println(io, "│  ├─ Hidden Sizes: $(setup.model_config.hidden_sizes)")
    println(io, "│  ├─ Activation: $(setup.model_config.activation)")
    println(io, "│  └─ Init Strategy: $(setup.model_config.init_strategy)")
    println(io, "├─ PDE Configuration:")
    println(io, "│  ├─ u_scale: $(setup.pde_config.u_scale)")
    println(io, "│  └─ tmax: $(setup.pde_config.rde_params.tmax)")
    println(io, "└─ Training Configuration:")
    println(io, "   ├─ Learning Rates: $(setup.training_config.learning_rates)")
    println(io, "   ├─ Iterations: $(setup.training_config.iterations)")
    println(io, "   ├─ Training Strategy: $(typeof(setup.training_config.training_strategy))")
    println(io, "   └─ Adaptive Loss: $(typeof(setup.training_config.adaptive_loss))")
end

"""
    Base.show(io::IO, ::MIME"text/plain", results::ExperimentResults)

Detailed display for ExperimentResults objects.
"""
function Base.show(io::IO, ::MIME"text/plain", results::ExperimentResults)
    println(io, "ExperimentResults:")
    println(io, "├─ Metrics:")
    println(io, "│  ├─ MSE: $(round(results.metrics.mse, digits=6))")
    println(io, "│  ├─ MAE: $(round(results.metrics.mae, digits=6))")
    println(io, "│  ├─ RMSE: $(round(results.metrics.rmse, digits=6))")
    println(io, "│  ├─ R²: $(round(results.metrics.r_squared, digits=6))")
    println(io, "│  ├─ Final Loss: $(round(results.metrics.final_loss, digits=6))")
    println(io, "│  └─ Training Time: $(round(results.metrics.training_time, digits=2)) seconds")
    println(io, "├─ Predictions: $(length(results.predictions)) variables")
    println(io, "│  └─ Keys: $(keys(results.predictions))")
    println(io, "└─ Timestamp: $(results.timestamp)")
end

"""
    create_experiment(name, pde_config, model_config, training_config)

Create a new experiment for Rotating Detonation Engine PINN modeling.
"""
function create_experiment(name, pde_config, model_config, training_config)
    return ExperimentSetup(name, pde_config, model_config, training_config)
end

"""
    run_experiment(setup::ExperimentSetup; skip_plots=false)

Run a Rotating Detonation Engine PINN experiment.
If skip_plots is true, plots will not be saved (useful when running as part of a sweep).
"""
function run_experiment(setup::ExperimentSetup; skip_plots=false)
    # Create temporary variables for @dict
    name = setup.name
    u_scale = setup.pde_config.u_scale
    hidden_sizes = setup.model_config.hidden_sizes
    iterations = setup.training_config.iterations[end]  # Use the final iterations value
    
    # Create a dictionary of parameters for DrWatson's produce_or_load
    parameters = @strdict name u_scale hidden_sizes iterations
    
    # Check if we've already run this experiment
    result_dict, file = produce_or_load(
        datadir("experiments"), 
        parameters; 
        suffix = "jld2",
        force = false
    ) do params
        # This function runs if the experiment doesn't exist yet
        @info "Creating new experiment: $(name)"
        
        # Create PDE system
        pde_system = create_pde_system(setup.pde_config)
        
        # Create neural networks (one for u and one for λ)
        chains = create_neural_network(setup.model_config, 2, 1)  # 2 inputs (t, x), 1 output per network
        
        # Train the model
        loss_history = Float64[]
        @info "Training neural network..." model_architecture=setup.model_config.hidden_sizes total_iterations=sum(setup.training_config.iterations)
        res, discretization, sym_prob, training_time = train_model(pde_system, chains, setup.training_config)
        @info "Training completed in $(round(training_time, digits=2)) seconds"
        minimizers = [res.u.depvar[sym_prob.depvars[i]] for i in 1:2]
        # Create prediction functions
        predict_u, predict_λ = create_prediction_functions(discretization, res, sym_prob)
        
        # Run simulation for comparison
        @info "Running simulation for comparison..."
        ts_sim, xs_sim, us_sim, λs_sim = run_simulation(setup.pde_config.rde_params)
        
        # Predict on grid
        @info "Generating predictions on simulation grid..."
        us, λs = predict_on_grid(predict_u, predict_λ, ts_sim, xs_sim)
        
        
        # Calculate metrics
        @info "Calculating metrics..."
        # Flatten predictions and targets for metric calculation
        predictions_flat = vcat([us[i] for i in 1:length(ts_sim)]...)
        targets_flat = vcat([us_sim[i] for i in 1:length(ts_sim)]...)
        
        metrics = calculate_metrics(predictions_flat, targets_flat, res.objective, loss_history, training_time)
        
        # Store predictions for visualization
        predictions = Dict(
            :ts => ts_sim,
            :xs => xs_sim,
            :us => us,
            :λs => λs,
            :us_sim => us_sim,
            :λs_sim => λs_sim,
        )
        
        # Return a dictionary with all the results
        @info "Experiment completed" mse=round(metrics.mse, digits=6) final_loss=round(metrics.final_loss, digits=6) training_time=round(training_time, digits=2)
        data_dict = merge(params, Dict(
            "metrics" => metrics,
            "predictions" => predictions,
            "minimizers" => minimizers,
            "timestamp" => now(),
            "mse" => metrics.mse,
            "final_loss" => metrics.final_loss,
            "mae" => metrics.mae,
            "rmse" => metrics.rmse,
            "r_squared" => metrics.r_squared,
            "training_time" => training_time,
            ))
        @show keys(data_dict)
        return data_dict
    end
    
    # Create ExperimentResults object
    @show keys(result_dict)
    results = ExperimentResults(result_dict)
    
    # Save plots unless skipped
    if !skip_plots
        save_experiment_plots(setup, results)
    end
    
    # Create result dictionary for easy access
    result_dict = Dict(
        "name" => setup.name,
        "setup" => setup,
        "results" => results,
        "metrics" => results.metrics
    )
    
    return result_dict
end

"""
    save_experiment_plots(setup::ExperimentSetup, results::ExperimentResults; custom_dir=nothing)

Save plots for a Rotating Detonation Engine experiment.
If custom_dir is provided, plots will be saved in that directory instead of the default location.
"""
function save_experiment_plots(setup::ExperimentSetup, results::ExperimentResults; custom_dir=nothing)
    # Create plot directory
    plot_dir = isnothing(custom_dir) ? plotsdir("experiments", setup.name) : custom_dir
    mkpath(plot_dir)
    
    @info "Generating plots for experiment: $(setup.name)"
    
    # Save plots    
    @info "Creating comparison plot..."
    fig = plot_comparison(results.predictions, 
        title="$(setup.name) - RDE PINN vs Simulation"
    )
    safesave(joinpath(plot_dir, "comparison.png"), fig)
    
    @info "Creating error analysis plot..."
    fig = plot_error(
        results.predictions[:ts],
        results.predictions[:xs],
        results.predictions[:us],
        results.predictions[:λs],
        results.predictions[:us_sim],
        results.predictions[:λs_sim],
        title="$(setup.name) - Error Analysis"
    )
    safesave(joinpath(plot_dir, "error_analysis.png"), fig)
    
    @info "Plots saved to: $(plot_dir)"
end

"""
    compare_experiments(result_dicts; metrics=["mse", "mae", "final_loss", "training_time"])

Compare multiple Rotating Detonation Engine experiments based on specified metrics.
If sweep_dir is provided, comparison plots will be saved in that directory.
"""
function compare_experiments(result_dicts; 
                            metrics=["mse", "mae", "final_loss", "training_time"],
                            sweep_dir=nothing)
    # Extract metrics from each experiment
    metrics_list = []
    names = []
    
    for dict in result_dicts
        push!(metrics_list, dict["metrics"])
        push!(names, dict["name"])
    end
    
    @info "Comparing metrics across $(length(result_dicts)) experiments" metrics=metrics
    comparison = compare_metrics(metrics_list, names)
    
    # Create comparison plots
    if isnothing(sweep_dir)
        plot_dir = plotsdir("comparisons")
    else
        # Create a metrics_plots directory within the sweep directory
        plot_dir = joinpath(sweep_dir, "metrics_plots")
    end
    mkpath(plot_dir)
    
    for metric in metrics
        @info "Creating comparison plot for $(metric)..."
        fig = plot_experiment_comparison(
            result_dicts,
            metric=metric,
            title="RDE Experiment Comparison - $metric"
        )
        safesave(joinpath(plot_dir, "comparison_$(metric).png"), fig)
    end
    
    @info "Comparison plots saved to: $(plot_dir)"
    
    return comparison
end

"""
    run_hyperparameter_sweep(base_setup::ExperimentSetup, param_name, param_values; experiment_name_prefix="sweep")

Run a hyperparameter sweep for Rotating Detonation Engine PINN models.
"""
function run_hyperparameter_sweep(base_setup::ExperimentSetup, param_name, param_values; experiment_name_prefix="sweep")
    result_dicts = []
    
    @info "Starting hyperparameter sweep" parameter=param_name values=param_values
    
    # Create sweep directory for plots - single folder for all plots
    sweep_plots_dir = plotsdir("sweeps", "$(experiment_name_prefix)_$(param_name)")
    mkpath(sweep_plots_dir)
    
    # Create specific folders for comparison and error analysis plots
    comparison_dir = joinpath(sweep_plots_dir, "comparison_plots")
    error_dir = joinpath(sweep_plots_dir, "error_analysis_plots")
    mkpath(comparison_dir)
    mkpath(error_dir)
    
    for (i, value) in enumerate(param_values)
        # Create a new experiment with the modified parameter
        experiment_name = "$(experiment_name_prefix)_$(param_name)_$(value)"
        
        if param_name == "hidden_sizes"
            # Create new model config with different hidden sizes
            model_config = ModelConfig(
                value,
                base_setup.model_config.activation,
                base_setup.model_config.init_strategy
            )
            setup = create_experiment(
                experiment_name,
                base_setup.pde_config,
                model_config,
                base_setup.training_config
            )
        elseif param_name == "u_scale"
            # Create new PDE config with different u_scale
            pde_config = PDESystemConfig(
                base_setup.pde_config.rde_params,
                value
            )
            setup = create_experiment(
                experiment_name,
                pde_config,
                base_setup.model_config,
                base_setup.training_config
            )
        elseif param_name == "iterations"
            # Create new training config with different iterations
            training_config = TrainingConfig(
                base_setup.training_config.optimizer,
                base_setup.training_config.learning_rates,
                value,
                base_setup.training_config.training_strategy,
                base_setup.training_config.adaptive_loss
            )
            setup = create_experiment(
                experiment_name,
                base_setup.pde_config,
                base_setup.model_config,
                training_config
            )
        elseif param_name == "training_strategy"
            # Create new training config with different training strategy
            training_config = TrainingConfig(
                base_setup.training_config.optimizer,
                base_setup.training_config.learning_rates,
                base_setup.training_config.iterations,
                value,
                base_setup.training_config.adaptive_loss
            )
            setup = create_experiment(
                experiment_name,
                base_setup.pde_config,
                base_setup.model_config,
                training_config
            )
        elseif param_name == "adaptive_loss"
            # Create new training config with different adaptive loss
            training_config = TrainingConfig(
                base_setup.training_config.optimizer,
                base_setup.training_config.learning_rates,
                base_setup.training_config.iterations,
                base_setup.training_config.training_strategy,
                value
            )
            setup = create_experiment(
                experiment_name,
                base_setup.pde_config,
                base_setup.model_config,
                training_config
            )
        else
            error("Unsupported parameter for sweep: $param_name")
        end
        
        # Run the experiment
        @info "Running experiment $(i)/$(length(param_values))" experiment_name=experiment_name parameter=param_name value=value
        
        # Run the experiment but don't save plots yet
        result_dict = run_experiment(setup, skip_plots=true)
        
        # Save plots directly to the appropriate folders with parameter value in filename
        # Create comparison plot
        @info "Creating comparison plot for parameter value: $(value)"
        fig = plot_comparison(result_dict["results"].predictions, 
            title="$(experiment_name) - RDE PINN vs Simulation"
        )
        safesave(joinpath(comparison_dir, "comparison_$(param_name)_$(value).png"), fig)
        
        # Create error analysis plot
        @info "Creating error analysis plot for parameter value: $(value)"
        fig = plot_error(
            result_dict["results"].predictions[:ts],
            result_dict["results"].predictions[:xs],
            result_dict["results"].predictions[:us],
            result_dict["results"].predictions[:λs],
            result_dict["results"].predictions[:us_sim],
            result_dict["results"].predictions[:λs_sim],
            title="$(experiment_name) - Error Analysis"
        )
        safesave(joinpath(error_dir, "error_analysis_$(param_name)_$(value).png"), fig)
        
        push!(result_dicts, result_dict)
    end
    
    # Compare experiments
    @info "Comparing experiment results..."
    comparison = compare_experiments(result_dicts, sweep_dir=sweep_plots_dir)
    
    # Save sweep results
    sweep_dir = datadir("sweeps", "$(experiment_name_prefix)_$(param_name)")
    mkpath(sweep_dir)
    
    # Save sweep metadata
    sweep_data = Dict(
        "param_name" => param_name,
        "param_values" => param_values,
        "experiment_names" => [dict["name"] for dict in result_dicts],
        "comparison" => comparison,
        "timestamp" => now()
    )
    
    tagsave(
        joinpath(sweep_dir, "sweep_metadata.jld2"),
        sweep_data;
        safe = true
    )
    
    @info "Hyperparameter sweep completed" sweep_dir=sweep_dir sweep_plots_dir=sweep_plots_dir
    @info "Plots saved to:" comparison_plots=comparison_dir error_plots=error_dir metrics_plots=joinpath(sweep_plots_dir, "metrics_plots")
    
    return result_dicts, comparison
end

"""
    print_experiment_summary(setup::ExperimentSetup)

Print a summary of a Rotating Detonation Engine PINN experiment setup.
"""
function print_experiment_summary(setup::ExperimentSetup)
    println("Experiment: $(setup.name)")
    println("----------------------------------------")
    println("Model Configuration:")
    println("  Hidden Sizes: $(setup.model_config.hidden_sizes)")
    println("  Activation: $(setup.model_config.activation)")
    println("  Init Strategy: $(setup.model_config.init_strategy)")
    println()
    
    println("PDE Configuration:")
    println("  u_scale: $(setup.pde_config.u_scale)")
    println("  tmax: $(setup.pde_config.rde_params.tmax)")
    println()
    
    println("Training Configuration:")
    println("  Learning Rates: $(setup.training_config.learning_rates)")
    println("  Iterations: $(setup.training_config.iterations)")
    println("  Training Strategy: $(typeof(setup.training_config.training_strategy))")
    println("  Adaptive Loss: $(typeof(setup.training_config.adaptive_loss))")
    println()
end

"""
    print_experiment_summary(setup::ExperimentSetup, results::ExperimentResults)

Print a summary of a Rotating Detonation Engine PINN experiment with results.
"""
function print_experiment_summary(setup::ExperimentSetup, results::ExperimentResults)
    print_experiment_summary(setup)
    
    println("Metrics:")
    println("  MSE: $(round(results.metrics.mse, digits=6))")
    println("  MAE: $(round(results.metrics.mae, digits=6))")
    println("  RMSE: $(round(results.metrics.rmse, digits=6))")
    println("  R²: $(round(results.metrics.r_squared, digits=6))")
    println("  Final Loss: $(round(results.metrics.final_loss, digits=6))")
    println()
    
    println("Timestamp: $(results.timestamp)")
    println()
end

"""
    print_experiment_summary(result_dict::Dict)

Print a summary of a Rotating Detonation Engine PINN experiment from a result dictionary.
"""
function print_experiment_summary(result_dict::Dict)
    # Create ExperimentSetup from dictionary
    setup = ExperimentSetup(
        result_dict["name"],
        result_dict["pde_config"],
        result_dict["model_config"],
        result_dict["training_config"]
    )
    
    # Create ExperimentResults from dictionary
    results = ExperimentResults(
        result_dict["metrics"],
        result_dict["predictions"],
        result_dict["minimizers"],
        result_dict["timestamp"]
    )
    
    print_experiment_summary(setup, results)
end

"""
    load_experiment(filename::String)

Load a Rotating Detonation Engine experiment from disk.
"""
function load_experiment(filename::String)
    # Load the data
    data = load(filename)
    
    # Create ExperimentSetup
    setup = ExperimentSetup(
        data["name"],
        data["pde_config"],
        data["model_config"],
        data["training_config"]
    )
    
    # Create ExperimentResults
    results = ExperimentResults(
        data["metrics"],
        data["predictions"],
        data["minimizers"],
        data["timestamp"]
    )
    
    return setup, results
end

"""
    ExperimentDisplay

A struct that combines ExperimentSetup and ExperimentResults for display purposes.
"""
struct ExperimentDisplay
    setup::ExperimentSetup
    results::ExperimentResults
end

"""
    Base.show(io::IO, display::ExperimentDisplay)

Custom display for ExperimentDisplay objects.
"""
function Base.show(io::IO, display::ExperimentDisplay)
    println(io, "Experiment: $(display.setup.name)")
    println(io, "├─ Model: $(length(display.setup.model_config.hidden_sizes)) layers $(display.setup.model_config.hidden_sizes)")
    println(io, "├─ PDE: u_scale=$(display.setup.pde_config.u_scale), tmax=$(display.setup.pde_config.rde_params.tmax)")
    println(io, "├─ Training: $(length(display.setup.training_config.iterations)) stages, $(sum(display.setup.training_config.iterations)) total iterations")
    println(io, "├─ MSE: $(round(display.results.metrics.mse, digits=6))")
    println(io, "├─ RMSE: $(round(display.results.metrics.rmse, digits=6))")
    println(io, "└─ Final Loss: $(round(display.results.metrics.final_loss, digits=6))")
end

"""
    Base.show(io::IO, ::MIME"text/plain", display::ExperimentDisplay)

Detailed display for ExperimentDisplay objects.
"""
function Base.show(io::IO, ::MIME"text/plain", display::ExperimentDisplay)
    println(io, "Experiment: $(display.setup.name)")
    println(io, "├─ Model Configuration:")
    println(io, "│  ├─ Hidden Sizes: $(display.setup.model_config.hidden_sizes)")
    println(io, "│  ├─ Activation: $(display.setup.model_config.activation)")
    println(io, "│  └─ Init Strategy: $(display.setup.model_config.init_strategy)")
    println(io, "├─ PDE Configuration:")
    println(io, "│  ├─ u_scale: $(display.setup.pde_config.u_scale)")
    println(io, "│  └─ tmax: $(display.setup.pde_config.rde_params.tmax)")
    println(io, "├─ Training Configuration:")
    println(io, "│  ├─ Learning Rates: $(display.setup.training_config.learning_rates)")
    println(io, "│  ├─ Iterations: $(display.setup.training_config.iterations)")
    println(io, "│  ├─ Training Strategy: $(typeof(display.setup.training_config.training_strategy))")
    println(io, "│  └─ Adaptive Loss: $(typeof(display.setup.training_config.adaptive_loss))")
    println(io, "├─ Metrics:")
    println(io, "│  ├─ MSE: $(round(display.results.metrics.mse, digits=6))")
    println(io, "│  ├─ MAE: $(round(display.results.metrics.mae, digits=6))")
    println(io, "│  ├─ RMSE: $(round(display.results.metrics.rmse, digits=6))")
    println(io, "│  ├─ R²: $(round(display.results.metrics.r_squared, digits=6))")
    println(io, "│  └─ Final Loss: $(round(display.results.metrics.final_loss, digits=6))")
    println(io, "├─ Predictions: $(length(display.results.predictions)) variables")
    println(io, "│  └─ Keys: $(keys(display.results.predictions))")
    println(io, "└─ Timestamp: $(display.results.timestamp)")
end

"""
    experiment_display(result_dict::Dict)

Create an ExperimentDisplay object from a result dictionary.
"""
function experiment_display(result_dict::Dict)
    # Create ExperimentSetup from dictionary
    setup = result_dict["setup"]
    
    # Create ExperimentResults from dictionary
    @show result_dict
    results = result_dict["results"]
    
    return ExperimentDisplay(setup, results)
end 