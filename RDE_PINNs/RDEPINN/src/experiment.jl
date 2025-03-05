using JLD2
using Dates
using DrWatson

"""
    Experiment

Structure to store information about a Rotating Detonation Engine (RDE) PINN experiment.
"""
struct Experiment
    name::String
    pde_config::PDESystemConfig
    model_config::ModelConfig
    training_config::TrainingConfig
    results::Any
    metrics::Metrics
    predictions::Dict{Symbol, Any}
end

"""
    create_experiment(name, pde_config, model_config, training_config)

Create a new experiment for Rotating Detonation Engine PINN modeling.
"""
function create_experiment(name, pde_config, model_config, training_config)
    # Create placeholder for results and metrics
    results = nothing
    metrics = Metrics(0.0, 0.0, 0.0, 0.0, 0.0, Dict{Symbol, Vector{Float64}}())
    predictions = Dict{Symbol, Any}()
    
    return Experiment(name, pde_config, model_config, training_config, results, metrics, predictions)
end

"""
    run_experiment(experiment::Experiment)

Run a Rotating Detonation Engine PINN experiment with the specified configuration.
"""
function run_experiment(experiment::Experiment)
    # Create temporary variables for @dict
    name = experiment.name
    u_scale = experiment.pde_config.u_scale
    hidden_sizes = experiment.model_config.hidden_sizes
    iterations = experiment.training_config.iterations
    
    # Create a dictionary of parameters for DrWatson's produce_or_load
    parameters = @dict name u_scale hidden_sizes iterations
    
    # Check if we've already run this experiment
    result_dict, exists = produce_or_load(
        datadir("experiments"), 
        parameters; 
        suffix = "jld2",
        force = false
    ) do params
        # This function runs if the experiment doesn't exist yet
        
        # Create PDE system
        pde_system = create_pde_system(experiment.pde_config)
        
        # Create neural networks (one for u and one for λ)
        chains = create_neural_network(experiment.model_config, 2, 1)  # 2 inputs (t, x), 1 output per network
        
        # Train the model
        loss_history = Float64[]
        res, discretization, sym_prob = train_model(pde_system, chains, experiment.training_config)
        
        # Create prediction functions
        predict_u, predict_λ = create_prediction_functions(discretization, res, sym_prob)
        
        # Generate grid for prediction
        domains = pde_system.domain
        ts = generate_grid([domains[1]], resolution=0.01)[1]
        xs = generate_grid([domains[2]], resolution=0.01)[1]
        
        # Predict on grid
        us, λs = predict_on_grid(predict_u, predict_λ, ts, xs)
        
        # Run simulation for comparison
        ts_sim, xs_sim, us_sim, λs_sim = run_simulation(experiment.pde_config.rde_params)
        
        # Calculate metrics
        # Flatten predictions and targets for metric calculation
        predictions_flat = vcat([us[i] for i in 1:length(ts)]...)
        targets_flat = vcat([us_sim[i] for i in 1:length(ts_sim)]...)
        
        metrics = calculate_metrics(predictions_flat, targets_flat, res.objective, loss_history)
        
        # Store results and predictions
        predictions = Dict{Symbol, Any}(
            :ts => ts,
            :xs => xs,
            :us => us,
            :λs => λs,
            :ts_sim => ts_sim,
            :xs_sim => xs_sim,
            :us_sim => us_sim,
            :λs_sim => λs_sim
        )
        
        # Return a dictionary with all the results
        return Dict(
            "experiment_name" => experiment.name,
            "pde_config" => experiment.pde_config,
            "model_config" => experiment.model_config,
            "training_config" => experiment.training_config,
            "results" => res,
            "metrics" => metrics,
            "predictions" => predictions,
            "timestamp" => now()
        )
    end
    
    if exists
        # If the experiment already exists, load it
        println("Loading existing experiment: $(experiment.name)")
        
        # Create updated experiment with loaded results
        updated_experiment = Experiment(
            result_dict["experiment_name"],
            result_dict["pde_config"],
            result_dict["model_config"],
            result_dict["training_config"],
            result_dict["results"],
            result_dict["metrics"],
            result_dict["predictions"]
        )
    else
        # If this is a new experiment, create the updated experiment
        println("Completed new experiment: $(experiment.name)")
        
        # Create updated experiment with results
        updated_experiment = Experiment(
            experiment.name,
            experiment.pde_config,
            experiment.model_config,
            experiment.training_config,
            result_dict["results"],
            result_dict["metrics"],
            result_dict["predictions"]
        )
        
        # Generate and save plots
        save_experiment_plots(updated_experiment)
    end
    
    return updated_experiment
end

"""
    save_experiment_plots(experiment::Experiment)

Save plots for a Rotating Detonation Engine experiment.
"""
function save_experiment_plots(experiment::Experiment)
    # Create plot directory
    plot_dir = plotsdir("experiments", experiment.name)
    mkpath(plot_dir)
    
    # Save plots
    fig = plot_solution(
        experiment.predictions[:ts],
        experiment.predictions[:xs],
        experiment.predictions[:us],
        experiment.predictions[:λs],
        title="$(experiment.name) - RDE PINN Solution"
    )
    safesave(joinpath(plot_dir, "solution.png"), fig)
    
    fig = plot_comparison(
        experiment.predictions[:ts],
        experiment.predictions[:xs],
        experiment.predictions[:us],
        experiment.predictions[:λs],
        experiment.predictions[:us_sim],
        experiment.predictions[:λs_sim],
        title="$(experiment.name) - RDE PINN vs Simulation"
    )
    safesave(joinpath(plot_dir, "comparison.png"), fig)
    
    fig = plot_error(
        experiment.predictions[:ts],
        experiment.predictions[:xs],
        experiment.predictions[:us],
        experiment.predictions[:λs],
        experiment.predictions[:us_sim],
        experiment.predictions[:λs_sim],
        title="$(experiment.name) - Error Analysis"
    )
    safesave(joinpath(plot_dir, "error.png"), fig)
    
    # Create and save animation
    create_animation(
        experiment.predictions[:ts],
        experiment.predictions[:xs],
        experiment.predictions[:us],
        experiment.predictions[:λs],
        filename=joinpath(plot_dir, "animation.mp4")
    )
    
    println("Experiment plots saved to $(plot_dir)")
    
    return plot_dir
end

"""
    save_experiment(experiment::Experiment)

Save a Rotating Detonation Engine experiment to disk using DrWatson.
"""
function save_experiment(experiment::Experiment)
    # Create a dictionary with all experiment data
    data = Dict(
        "experiment_name" => experiment.name,
        "pde_config" => experiment.pde_config,
        "model_config" => experiment.model_config,
        "training_config" => experiment.training_config,
        "results" => experiment.results,
        "metrics" => experiment.metrics,
        "predictions" => experiment.predictions,
        "timestamp" => now()
    )
    
    # Create temporary variables for @dict
    name = experiment.name
    u_scale = experiment.pde_config.u_scale
    hidden_sizes = experiment.model_config.hidden_sizes
    iterations = experiment.training_config.iterations
    
    # Create parameters dictionary for filename
    parameters = @dict name u_scale hidden_sizes iterations
    
    # Save with DrWatson's tagsave
    experiment_file = tagsave(
        datadir("experiments", savename(parameters, "jld2")),
        data;
        safe = true
    )
    
    # Generate and save plots
    plot_dir = save_experiment_plots(experiment)
    
    println("Experiment saved to $(experiment_file)")
    
    return experiment_file
end

"""
    load_experiment(name)

Load a Rotating Detonation Engine experiment from disk using DrWatson.
"""
function load_experiment(name)
    # Find all experiment files
    experiment_files = readdir(datadir("experiments"))
    
    # Find the file that matches the experiment name
    experiment_file = nothing
    for file in experiment_files
        if occursin(name, file) && endswith(file, ".jld2")
            experiment_file = file
            break
        end
    end
    
    if experiment_file === nothing
        error("Experiment '$(name)' not found in $(datadir("experiments"))")
    end
    
    # Load the experiment data
    data = load(datadir("experiments", experiment_file))
    
    # Create experiment
    experiment = Experiment(
        data["experiment_name"],
        data["pde_config"],
        data["model_config"],
        data["training_config"],
        data["results"],
        data["metrics"],
        data["predictions"]
    )
    
    return experiment
end

"""
    compare_experiments(experiments; metrics=["mse", "final_losERROR: ArgumentError: Invalid field syntax
Stacktrace:
 [1] run_experiment(experiment::Experiment)
   @ Main.RDEPINN ~/Code/FYS9429/RDE_PINNs/src/experiment.jl:41
 [2] top-level scope
   @ ~/Code/FYS9429/RDE_PINNs/scripts/rde_pinn_example.jl:34s"])

Compare multiple Rotating Detonation Engine experiments.
"""
function compare_experiments(experiments; metrics=["mse", "final_loss"])
    names = [exp.name for exp in experiments]
    metrics_list = [exp.metrics for exp in experiments]
    
    # Compare metrics
    comparison = compare_metrics(metrics_list, names)
    
    # Create comparison plots
    plot_dir = plotsdir("comparisons")
    mkpath(plot_dir)
    
    for metric in metrics
        fig = plot_experiment_comparison(
            experiments,
            metric=metric,
            title="RDE Experiment Comparison - $metric"
        )
        safesave(joinpath(plot_dir, "comparison_$(metric).png"), fig)
    end
    
    println("Comparison plots saved to $(plot_dir)")
    
    return comparison
end

"""
    run_hyperparameter_sweep(base_experiment, param_name, param_values; experiment_name_prefix="sweep")

Run a hyperparameter sweep for Rotating Detonation Engine PINN models.
"""
function run_hyperparameter_sweep(base_experiment, param_name, param_values; experiment_name_prefix="sweep")
    experiments = []
    
    for (i, value) in enumerate(param_values)
        # Create a new experiment with the modified parameter
        experiment_name = "$(experiment_name_prefix)_$(param_name)_$(i)"
        
        if param_name == "hidden_sizes"
            # Create new model config with different hidden sizes
            model_config = ModelConfig(
                value,
                base_experiment.model_config.activation,
                base_experiment.model_config.init_strategy
            )
            experiment = create_experiment(
                experiment_name,
                base_experiment.pde_config,
                model_config,
                base_experiment.training_config
            )
        elseif param_name == "u_scale"
            # Create new PDE config with different u_scale
            pde_config = PDESystemConfig(
                base_experiment.pde_config.rde_params,
                value,
                base_experiment.pde_config.periodic_boundary
            )
            experiment = create_experiment(
                experiment_name,
                pde_config,
                base_experiment.model_config,
                base_experiment.training_config
            )
        elseif param_name == "iterations"
            # Create new training config with different iterations
            training_config = TrainingConfig(
                base_experiment.training_config.optimizer,
                value,
                base_experiment.training_config.batch_size,
                base_experiment.training_config.training_strategy,
                base_experiment.training_config.adaptive_loss,
                base_experiment.training_config.print_interval,
                base_experiment.training_config.callback_interval
            )
            experiment = create_experiment(
                experiment_name,
                base_experiment.pde_config,
                base_experiment.model_config,
                training_config
            )
        elseif param_name == "training_strategy"
            # Create new training config with different training strategy
            training_config = TrainingConfig(
                base_experiment.training_config.optimizer,
                base_experiment.training_config.iterations,
                base_experiment.training_config.batch_size,
                value,
                base_experiment.training_config.adaptive_loss,
                base_experiment.training_config.print_interval,
                base_experiment.training_config.callback_interval
            )
            experiment = create_experiment(
                experiment_name,
                base_experiment.pde_config,
                base_experiment.model_config,
                training_config
            )
        elseif param_name == "adaptive_loss"
            # Create new training config with different adaptive loss
            training_config = TrainingConfig(
                base_experiment.training_config.optimizer,
                base_experiment.training_config.iterations,
                base_experiment.training_config.batch_size,
                base_experiment.training_config.training_strategy,
                value,
                base_experiment.training_config.print_interval,
                base_experiment.training_config.callback_interval
            )
            experiment = create_experiment(
                experiment_name,
                base_experiment.pde_config,
                base_experiment.model_config,
                training_config
            )
        else
            error("Unsupported parameter for sweep: $param_name")
        end
        
        # Run the experiment
        println("Running experiment $experiment_name with $param_name = $value")
        experiment = run_experiment(experiment)
        
        push!(experiments, experiment)
    end
    
    # Compare experiments
    comparison = compare_experiments(experiments)
    
    # Save sweep results
    sweep_dir = datadir("sweeps", "$(experiment_name_prefix)_$(param_name)")
    mkpath(sweep_dir)
    
    # Save sweep metadata
    sweep_data = Dict(
        "param_name" => param_name,
        "param_values" => param_values,
        "experiment_names" => [exp.name for exp in experiments],
        "comparison" => comparison,
        "timestamp" => now()
    )
    
    tagsave(
        joinpath(sweep_dir, "sweep_metadata.jld2"),
        sweep_data;
        safe = true
    )
    
    println("Hyperparameter sweep saved to $(sweep_dir)")
    
    return experiments, comparison
end

"""
    print_experiment_summary(experiment::Experiment)

Print a summary of a Rotating Detonation Engine PINN experiment.
"""
function print_experiment_summary(experiment::Experiment)
    println("Experiment: $(experiment.name)")
    println("----------------------------------------")
    println("Model Configuration:")
    println("  Hidden Sizes: $(experiment.model_config.hidden_sizes)")
    println("  Activation: $(experiment.model_config.activation)")
    println("  Init Strategy: $(experiment.model_config.init_strategy)")
    println()
    
    println("PDE Configuration:")
    println("  u_scale: $(experiment.pde_config.u_scale)")
    println("  Periodic Boundary: $(experiment.pde_config.periodic_boundary)")
    println("  tmax: $(experiment.pde_config.rde_params.tmax)")
    println()
    
    println("Training Configuration:")
    println("  Iterations: $(experiment.training_config.iterations)")
    println("  Batch Size: $(experiment.training_config.batch_size)")
    println("  Training Strategy: $(experiment.training_config.training_strategy)")
    println("  Adaptive Loss: $(experiment.training_config.adaptive_loss)")
    println()
    
    println("Metrics:")
    println("  MSE: $(round(experiment.metrics.mse, digits=6))")
    println("  MAE: $(round(experiment.metrics.mae, digits=6))")
    println("  RMSE: $(round(experiment.metrics.rmse, digits=6))")
    println("  R²: $(round(experiment.metrics.r_squared, digits=6))")
    println("  Final Loss: $(round(experiment.metrics.final_loss, digits=6))")
    println()
end 