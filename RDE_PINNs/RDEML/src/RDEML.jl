__precompile__(false)
module RDEML
"""
RDEML - Rotating Detonation Engine Machine Learning

This module provides a framework for solving Rotating Detonation Engine (RDE) 
models using Machine Learning.
"""

    using CairoMakie
    using Dates
    using DrWatson
    using DRL_RDE_utils
    using JLD2
    using LineSearches
    using Lux
    using ModelingToolkit
    using ModelingToolkit: Interval, infimum, supremum
    using NeuralOperators
    using NeuralPDE
    using Optimization
    using OptimizationOptimJL
    using OptimizationOptimisers
    using ProgressMeter
    using Random
    using RDE
    using RDE_Env
    using Statistics

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
    export make_data_policies_and_envs, make_data_reset_strategies,
        collect_data, save_data

    include("fnos/fno.jl")
    export train!, FNO, FNOConfig

    include("fnos/visualization.jl")
    export plot_losses, visualize_data, plot_test_comparison

    include("fnos/analysis.jl")
    export compare_to_policy

    include("fnos/model_io.jl")
    export save_fno_config, load_fno_config, list_saved_models, 
       produce_or_load_fno, save_prediction, save_fno_config_direct, load_fno_config_direct 

end # module 