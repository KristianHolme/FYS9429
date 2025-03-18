"""
    TrainingConfig

Configuration for training Physics-Informed Neural Networks for Rotating Detonation Engine (RDE) simulations.
"""
struct TrainingConfig
    optimizer::Any
    learning_rates::Vector{Float64}
    iterations::Vector{Int}
    training_strategy::NeuralPDE.AbstractTrainingStrategy
    adaptive_loss::Union{NeuralPDE.AbstractAdaptiveLoss, Nothing}
end

"""
    default_training_config(; 
        optimizer=OptimizationOptimisers.Adam, 
        learning_rates=[0.01, 0.001], 
        iterations=[1000, 2000], 
        training_strategy=QuasiRandomTraining(128),
        adaptive_loss=MiniMaxAdaptiveLoss(64))

Create a default training configuration for RDE PINN models.
"""
function default_training_config(; 
    optimizer=OptimizationOptimisers.Adam, 
    learning_rates=[0.01, 0.001], 
    iterations=[1000, 2000], 
    training_strategy=QuasiRandomTraining(128),
    adaptive_loss=MiniMaxAdaptiveLoss(64))
    
    return TrainingConfig(
        optimizer, 
        learning_rates, 
        iterations, 
        training_strategy, 
        adaptive_loss
    )
end

"""
    Base.show(io::IO, config::TrainingConfig)

Custom display for TrainingConfig objects.
"""
function Base.show(io::IO, config::TrainingConfig)
    println(io, "TrainingConfig:")
    println(io, "├─ Optimizer: $(config.optimizer)")
    println(io, "├─ Learning Rates: $(config.learning_rates)")
    println(io, "├─ Iterations: $(config.iterations) (total: $(sum(config.iterations)))")
    println(io, "├─ Training Strategy: $(typeof(config.training_strategy))")
    println(io, "└─ Adaptive Loss: $(typeof(config.adaptive_loss))")
end

"""
    Base.show(io::IO, ::MIME"text/plain", config::TrainingConfig)

Detailed display for TrainingConfig objects.
"""
function Base.show(io::IO, ::MIME"text/plain", config::TrainingConfig)
    println(io, "TrainingConfig:")
    println(io, "├─ Optimizer: $(config.optimizer)")
    println(io, "├─ Learning Rates: $(config.learning_rates)")
    println(io, "├─ Iterations: $(config.iterations)")
    println(io, "├─ Total Iterations: $(sum(config.iterations))")
    println(io, "├─ Training Strategy: $(typeof(config.training_strategy))")
    println(io, "└─ Adaptive Loss: $(typeof(config.adaptive_loss))")
end

"""
    fast_training_config()

Create a training configuration for quick testing of RDE PINNs.
"""
function fast_training_config()
    return TrainingConfig(
        OptimizationOptimisers.Adam, 
        [0.01, 0.001], 
        [500, 800], 
        QuasiRandomTraining(64), 
        MiniMaxAdaptiveLoss(64)
    )
end

"""
    medium_training_config()

Create a medium training configuration for RDE PINNs.
"""
function medium_training_config()
    return TrainingConfig(
        OptimizationOptimisers.Adam, 
        [0.01, 0.001], 
        [1000, 2000], 
        QuasiRandomTraining(128), 
        MiniMaxAdaptiveLoss(64)
    )
end

"""
    thorough_training_config()

Create a thorough training configuration for RDE PINNs with two-stage training.
"""
function thorough_training_config()
    return TrainingConfig(
        OptimizationOptimisers.Adam, 
        [0.01, 3e-4], 
        [2000, 4000], 
        QuasiRandomTraining(128), 
        MiniMaxAdaptiveLoss(64)
    )
end

"""
    long_training_config()

Create a training configuration for extensive RDE PINN training.
"""
function long_training_config()
    return TrainingConfig(
        OptimizationOptimisers.Adam, 
        [0.01, 3e-4], 
        [5000, 10000], 
        QuasiRandomTraining(128), 
        MiniMaxAdaptiveLoss(64)
    )
end

"""
    train_model(pde_system, model, config::TrainingConfig)

Train a PINN model for Rotating Detonation Engine simulation using the specified configuration.
"""
function train_model(pde_system, model, config::TrainingConfig)
    @assert length(config.learning_rates) == length(config.iterations) "Learning rates and iterations must have the same length"
    # Create discretization with appropriate strategy and adaptive loss
    discretization = PhysicsInformedNN(model, config.training_strategy, adaptive_loss=config.adaptive_loss)
    
    prob = discretize(pde_system, discretization)
    # Create symbolic problem
    sym_prob = symbolic_discretize(pde_system, discretization)
    
    # Get inner loss functions for progress tracking
    pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
    bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions
    
    # Create progress bar
    progressbar = Progress(sum(config.iterations), "Training...", showspeed=true)
    
    # Create callback function with progress bar
    callback = function (p, l)
        next!(progressbar, showvalues=[
            (:loss, l), 
            (:pde_losses, map(l_ -> l_(p.u), pde_inner_loss_functions)), 
            (:bcs_losses, map(l_ -> l_(p.u), bcs_inner_loss_functions))
        ])
        return false
    end
    # Create training problem

    # Start timing the training process
    training_start_time = time()

    opt = config.optimizer(config.learning_rates[1])
    res = solve(prob, opt; maxiters = config.iterations[1], callback)
    for i in 2:length(config.learning_rates)
        opt = config.optimizer(config.learning_rates[i])
        prob = remake(prob, u0=res.u)
        res = solve(prob, opt; maxiters = config.iterations[i], callback)
    end

    # Calculate total training time
    training_time = time() - training_start_time

    finish!(progressbar)
    
    return res, discretization, sym_prob, training_time
end

"""
    create_prediction_functions(discretization, res, sym_prob)

Create prediction functions for u and λ from trained model.
"""
function create_prediction_functions(discretization, res, sym_prob)
    phi = discretization.phi
    minimizers_ = [res.u.depvar[sym_prob.depvars[i]] for i in 1:2]
    
    predict_u(t, x) = phi[1]([t, x], minimizers_[1])[1]
    predict_λ(t, x) = phi[2]([t, x], minimizers_[2])[1]
    
    return predict_u, predict_λ
end

"""
    predict_on_grid(predict_u, predict_λ, ts, xs)

Generate predictions on a grid of points.
"""
function predict_on_grid(predict_u, predict_λ, ts, xs)
    us = [predict_u.(t, xs) for t in ts]
    λs = [predict_λ.(t, xs) for t in ts]
    return us, λs
end

"""
    generate_grid(domains; resolution=0.01)

Generate a grid of points for prediction.
"""
function generate_grid(domains; resolution=0.01)
    return [infimum(d.domain):resolution:supremum(d.domain) for d in domains]
end

"""
    TrainingResult

Structure to hold training results.
"""
struct TrainingResult
    result::Any
    sym_prob::Any
    discretization::Any
    ts::Vector{Float64}
    xs::Vector{Float64}
    us::Vector{Vector{Float64}}
    λs::Vector{Vector{Float64}}
    pde_losses::Vector{Float64}
    bc_losses::Vector{Float64}
    training_time::Float64
end

"""
    Base.show(io::IO, result::TrainingResult)

Custom display for TrainingResult objects.
"""
function Base.show(io::IO, result::TrainingResult)
    println(io, "TrainingResult:")
    println(io, "├─ Grid: $(length(result.ts)) time points × $(length(result.xs)) spatial points")
    println(io, "├─ PDE Losses: $(round.(result.pde_losses, digits=6))")
    println(io, "├─ BC Losses: $(round.(result.bc_losses, digits=6))")
    println(io, "└─ Training Time: $(round(result.training_time, digits=2)) seconds")
end

"""
    Base.show(io::IO, ::MIME"text/plain", result::TrainingResult)

Detailed display for TrainingResult objects.
"""
function Base.show(io::IO, ::MIME"text/plain", result::TrainingResult)
    println(io, "TrainingResult:")
    println(io, "├─ Time Domain: [$(minimum(result.ts)), $(maximum(result.ts))]")
    println(io, "├─ Space Domain: [$(minimum(result.xs)), $(maximum(result.xs))]")
    println(io, "├─ Grid: $(length(result.ts)) time points × $(length(result.xs)) spatial points")
    println(io, "├─ PDE Losses: $(round.(result.pde_losses, digits=6))")
    println(io, "├─ BC Losses: $(round.(result.bc_losses, digits=6))")
    println(io, "└─ Training Time: $(round(result.training_time, digits=2)) seconds")
end

"""
    train_and_predict(pdesystem, model_config::ModelConfig, training_config::TrainingConfig)

Train a model and generate predictions.
"""
function train_and_predict(pdesystem, model_config::ModelConfig, training_config::TrainingConfig)
    # Create neural networks (one for u and one for λ)
    chains = create_neural_network(model_config, 2, 1)  # 2 inputs (t, x), 1 output per network
    
    # Train model
    res, discretization, sym_prob, training_time = train_model(pdesystem, chains, training_config)
    
    # Create prediction functions
    predict_u, predict_λ = create_prediction_functions(discretization, res, sym_prob)
    
    # Generate grid and predictions
    domains = pdesystem.domain
    ts = generate_grid([domains[1]], resolution=0.01)[1]
    xs = generate_grid([domains[2]], resolution=0.01)[1]
    
    # Predict on grid
    us, λs = predict_on_grid(predict_u, predict_λ, ts, xs)
    
    # Calculate final losses
    pde_losses = [l(res.u) for l in sym_prob.loss_functions.pde_loss_functions]
    bc_losses = [l(res.u) for l in sym_prob.loss_functions.bc_loss_functions]
    
    # Return results
    return TrainingResult(
        res,
        sym_prob,
        discretization,
        ts,
        xs,
        us,
        λs,
        pde_losses,
        bc_losses,
        training_time
    )
end 