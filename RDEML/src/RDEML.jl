__precompile__(false)
module RDEML
    using Reexport
    using CairoMakie
    using Colors
    using DataFrames
    using Dates
    using DrWatson
    @reexport using DRL_RDE_utils
    using JLD2
    using LineSearches
    @reexport using Lux
    @reexport using LuxCUDA
    using ModelingToolkit
    using ModelingToolkit: Interval, infimum, supremum
    using NeuralOperators
    using NeuralPDE
    using Optimization
    using OptimizationOptimJL
    using OptimizationOptimisers
    using ProgressMeter
    @reexport using Random
    using Statistics
    using MLUtils

    ##PINN functions
    # Include submodules
    include("pinns/models.jl")
    include("pinns/pde_system.jl")
    include("pinns/training.jl")
    include("pinns/visualization.jl")
    include("pinns/metrics.jl")
    include("pinns/experiment.jl")

    # Export main functionality
    export PDESystemConfig, ModelConfig, TrainingConfig, Metrics
    export ExperimentSetup, ExperimentResults, ExperimentDisplay
    export create_pde_system, create_neural_network, train_model
    export create_prediction_functions, predict_on_grid, generate_grid, run_simulation

    # Export experiment functions
    export create_experiment, run_experiment, print_experiment_summary, experiment_display
    export save_experiment_plots, run_hyperparameter_sweep
    export save_experiment, load_experiment, compare_experiments

    # Export configuration functions
    export default_pde_config, default_model_config, default_training_config
    export small_model_config, medium_model_config, large_model_config
    export deep_model_config, tanh_model_config
    export fast_training_config, medium_training_config, thorough_training_config, long_training_config

    # Export visualization functions
    export plot_solution, plot_spacetime_slice, plot_error, plot_comparison
    export plot_experiment_comparison, create_animation

    # Export metrics functions
    export calculate_metrics, compare_metrics

    ##FNO functions
    include("fnos/data_gathering.jl")
    export get_data_policies, get_data_reset_strategies,
    collect_data, save_data, DataGatherer, DataSetInfo,
    prepare_dataset, generate_data, shuffle_batches!,
    number_of_samples, calculate_number_of_batches,
    FNODataset, create_dataloader
    
    include("fnos/fno.jl")
    export train!, FNO, FNOConfig, number_of_batches
    
    include("fnos/utils.jl")
    export train_and_save!, moving_average
    include("fnos/visualization.jl")
    export plot_losses, visualize_data, plot_test_comparison, 
        plot_parameter_analysis, plot_training_time_analysis, plot_losses_final_eval

    include("fnos/analysis.jl")
    export compare_to_policy, replace_sim_with_prediction, plot_initial_conditions,
        one_step_prediction, recursive_prediction, evaluate_operators

    include("fnos/model_io.jl")
    export fnoconfig_to_dict, dict_to_fnoconfig


end # module 