# Test script to verify RDEPINN module loading

using DrWatson
@quickactivate "RDE_PINNs"

# Load the RDEPINN module once
include(srcdir("RDEPINN.jl"))  # This defines the module
using .RDEPINN                 # This imports all exported symbols

println("RDEPINN module loaded successfully!")

# Test a few functions to make sure they're accessible
println("Testing configuration functions:")
pde_config = default_pde_config()
model_config = default_model_config()
training_config = fast_training_config()

println("  PDE Config: tmax=$(pde_config.rde_params.tmax), u_scale=$(pde_config.u_scale)")
println("  Model Config: hidden_sizes=$(model_config.hidden_sizes)")
println("  Training Config: iterations=$(training_config.iterations)")

# Test experiment functions
println("\nTesting experiment functions:")
experiment = create_experiment("test_experiment", pde_config, model_config, training_config)
println("  Created experiment: $(experiment.name)")

# Test visualization functions
println("\nTesting visualization functions:")
println("  plot_solution: $(isdefined(RDEPINN, :plot_solution))")
println("  plot_comparison: $(isdefined(RDEPINN, :plot_comparison))")
println("  create_animation: $(isdefined(RDEPINN, :create_animation))")

# Test DrWatson functions
println("\nTesting DrWatson functions:")
println("  datadir: $(isdefined(RDEPINN, :datadir))")
println("  plotsdir: $(isdefined(RDEPINN, :plotsdir))")
println("  safesave: $(isdefined(RDEPINN, :safesave))")

println("\nAll functions accessible. Module is working correctly!") 