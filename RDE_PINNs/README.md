# RDEPINN: Physics-Informed Neural Networks for Rotating Detonation Engines

This project provides a framework for solving Rotating Detonation Engine (RDE) equations using Physics-Informed Neural Networks (PINNs).

## Project Structure

The project is organized using DrWatson for reproducible scientific research:

```
RDE_PINNs/
├── data/               # Saved experiment data
│   ├── experiments/    # Individual experiment results
│   └── sweeps/         # Hyperparameter sweep results
├── plots/              # Generated plots and visualizations
│   ├── experiments/    # Plots for individual experiments
│   └── comparisons/    # Comparison plots
├── scripts/            # Example scripts
├── src/                # Source code
│   └── RDEPINN.jl      # Main module
└── _research/          # Research notes and documentation
```

## Getting Started

### Prerequisites

- Julia 1.6 or higher
- Required packages: NeuralPDE, Lux, ModelingToolkit, Optimization, CairoMakie, DrWatson, etc.

### Installation

1. Clone this repository
2. Navigate to the project directory
3. Start Julia and activate the project environment:

```julia
using DrWatson
@quickactivate "RDE_PINNs"
```

### Using the RDEPINN Module

To use the RDEPINN module in your scripts:

```julia
using DrWatson
@quickactivate "RDE_PINNs"

# Include the module and import all exported symbols
include(srcdir("RDEPINN.jl"))
using .RDEPINN
```

Then use the exported functions directly:

```julia
# Create configurations
pde_config = default_pde_config(tmax=1.0, u_scale=1.5)
model_config = default_model_config(hidden_sizes=[32, 32, 32])
training_config = fast_training_config()

# Create and run an experiment
experiment = create_experiment("my_experiment", pde_config, model_config, training_config)
experiment = run_experiment(experiment)
```

## Running Examples

The `scripts/` directory contains example scripts demonstrating how to use the framework:

- `rde_pinn_example.jl`: Basic example of solving RDE equations with PINNs
- `test_module.jl`: Simple script to verify the module is working correctly

To run an example:

```bash
julia scripts/rde_pinn_example.jl
```

## Creating and Running Experiments

The framework provides a simple API for creating and running experiments:

```julia
# Create configurations
pde_config = default_pde_config(tmax=1.0, u_scale=1.5)
model_config = default_model_config(hidden_sizes=[32, 32, 32])
training_config = fast_training_config()

# Create and run an experiment
experiment = create_experiment("my_experiment", pde_config, model_config, training_config)
experiment = run_experiment(experiment)

# Print experiment summary
print_experiment_summary(experiment)
```

## Hyperparameter Sweeps

The framework supports hyperparameter sweeps:

```julia
# Run a hyperparameter sweep on hidden layer sizes
hidden_sizes_values = [[16, 16], [32, 32, 32], [64, 64, 64, 64]]
experiments, comparison = run_hyperparameter_sweep(
    experiment, 
    "hidden_sizes", 
    hidden_sizes_values, 
    experiment_name_prefix="hidden_size_sweep"
)
```

## Data Management

All experiment data is automatically saved using DrWatson's data management tools:

- Experiment data is saved to `data/experiments/`
- Plots are saved to `plots/experiments/`
- Hyperparameter sweeps are saved to `data/sweeps/`

## Future Improvements

For better module management and type stability, consider converting RDEPINN to a proper Julia package with:

```
using Pkg
Pkg.generate("RDEPINN")  # Creates the package structure
```

Then move the module code to the new package structure and use:

```
using Pkg
Pkg.develop(path="/path/to/RDEPINN")  # Adds the package to your environment
```

This would allow you to simply import it with `using RDEPINN`.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 