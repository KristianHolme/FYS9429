module RDEPINN
"""
RDEPINN - Physics-Informed Neural Networks for Rotating Detonation Engines

This module provides a framework for solving Rotating Detonation Engine (RDE) 
equations using Physics-Informed Neural Networks (PINNs).
"""

using CairoMakie
using Dates
using DrWatson
using JLD2
using LineSearches
using Lux
using ModelingToolkit
using ModelingToolkit: Interval, infimum, supremum
using NeuralPDE
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using ProgressMeter
using Random
using RDE
using RDE_Env
using Statistics

# Include submodules
include("models.jl")
include("pde_system.jl")
include("training.jl")
include("visualization.jl")
include("metrics.jl")
include("experiment.jl")

# Export main functionality
export PDESystemConfig, ModelConfig, TrainingConfig, Metrics, Experiment
export create_pde_system, create_neural_network, train_model
export create_prediction_functions, predict_on_grid, generate_grid, run_simulation

# Export experiment functions
export create_experiment, run_experiment, print_experiment_summary
export save_experiment_plots, run_hyperparameter_sweep
export save_experiment, load_experiment, compare_experiments

# Export configuration functions
export default_pde_config, default_model_config, default_training_config
export small_model_config, medium_model_config, large_model_config
export deep_model_config, tanh_model_config
export fast_training_config, medium_training_config, thorough_training_config, long_training_config

# Export visualization functions
export plot_solution, plot_spacetime_slice, plot_error, plot_comparison
export plot_metrics_over_time, plot_experiment_comparison, create_animation

# Export metrics functions
export calculate_metrics, compare_metrics

# Export DrWatson functionality
export @quickactivate, projectdir, datadir, plotsdir, scriptsdir, srcdir
export safesave, produce_or_load, tagsave, @dict, dict_list

end # module 