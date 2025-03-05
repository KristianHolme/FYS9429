"""
    TrainingConfig

Configuration for training Physics-Informed Neural Networks for Rotating Detonation Engine (RDE) simulations.
"""
struct TrainingConfig
    optimizer::Any
    iterations::Int
    batch_size::Int
    training_strategy::Symbol  # :quasi_random or :stochastic
    adaptive_loss::Symbol      # :minimax or :gradient_scale
    print_interval::Int
    callback_interval::Int
end

"""
    default_training_config(; 
        optimizer=OptimizationOptimisers.Adam(0.01), 
        iterations=1000, 
        batch_size=128, 
        training_strategy=:quasi_random,
        adaptive_loss=:minimax,
        print_interval=100, 
        callback_interval=100)

Create a default training configuration for RDE PINN models.
"""
function default_training_config(; 
    optimizer=OptimizationOptimisers.Adam(0.01), 
    iterations=1000, 
    batch_size=128, 
    training_strategy=:quasi_random,
    adaptive_loss=:minimax,
    print_interval=100, 
    callback_interval=100)
    
    return TrainingConfig(
        optimizer, 
        iterations, 
        batch_size, 
        training_strategy, 
        adaptive_loss,
        print_interval, 
        callback_interval
    )
end

"""
    fast_training_config()

Create a training configuration for quick testing of RDE PINNs.
"""
function fast_training_config()
    return TrainingConfig(
        OptimizationOptimisers.Adam(0.01), 
        100, 
        64, 
        :quasi_random, 
        :minimax,
        10, 
        10
    )
end

"""
    medium_training_config()

Create a medium training configuration for RDE PINNs.
"""
function medium_training_config()
    return TrainingConfig(
        OptimizationOptimisers.Adam(0.01), 
        1000, 
        128, 
        :quasi_random, 
        :minimax,
        100, 
        100
    )
end

"""
    thorough_training_config()

Create a thorough training configuration for RDE PINNs with two-stage training.
"""
function thorough_training_config()
    return TrainingConfig(
        OptimizationOptimisers.Adam(0.01), 
        3000, 
        128, 
        :quasi_random, 
        :minimax,
        100, 
        100
    )
end

"""
    long_training_config()

Create a training configuration for extensive RDE PINN training.
"""
function long_training_config()
    return TrainingConfig(
        OptimizationOptimisers.Adam(0.005), 
        5000, 
        128, 
        :quasi_random, 
        :minimax,
        500, 
        500
    )
end

"""
    train_model(pde_system, model, config::TrainingConfig)

Train a PINN model for Rotating Detonation Engine simulation using the specified configuration.
"""
function train_model(pde_system, model, config::TrainingConfig)
    # Create training strategy based on configuration
    if config.training_strategy == :quasi_random
        strategy = QuasiRandomTraining(config.batch_size)
    elseif config.training_strategy == :stochastic
        strategy = StochasticTraining(config.batch_size)
    else
        @warn "Unknown training strategy: $(config.training_strategy). Using QuasiRandomTraining."
        strategy = QuasiRandomTraining(config.batch_size)
    end
    
    # Create adaptive loss based on configuration
    if config.adaptive_loss == :minimax
        adaptive_loss = MiniMaxAdaptiveLoss(config.batch_size)
    elseif config.adaptive_loss == :gradient_scale
        adaptive_loss = GradientScaleAdaptiveLoss(config.batch_size)
    else
        @warn "Unknown adaptive loss: $(config.adaptive_loss). Using MiniMaxAdaptiveLoss."
        adaptive_loss = MiniMaxAdaptiveLoss(config.batch_size)
    end
    
    # Create discretization with appropriate strategy and adaptive loss
    discretization = PhysicsInformedNN(model, strategy, adaptive_loss=adaptive_loss)
    
    # Create symbolic problem
    sym_prob = symbolic_discretize(pde_system, discretization)
    
    # Get inner loss functions for progress tracking
    pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
    bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions
    
    # Create progress bar
    progressbar = ProgressUnknown("Training...", showspeed=true)
    
    # Create callback function with progress bar
    callback = function (p, l)
        next!(progressbar, showvalues=[
            (:loss, l), 
            (:pde_losses, map(l_ -> l_(p.u), pde_inner_loss_functions)), 
            (:bcs_losses, map(l_ -> l_(p.u), bcs_inner_loss_functions))
        ])
        return false
    end
    
    # Create loss function
    loss = build_loss_function(discretization, sym_prob)
    
    # Create training problem
    prob = OptimizationProblem(loss, discretization.initial_params)
    
    # Train the model
    res = solve(prob, config.optimizer, maxiters=config.iterations, cb=callback)
    
    # If thorough training is requested, do a second stage with lower learning rate
    if config.iterations >= 3000
        @info "First stage training completed with loss = $(res.objective)"
        @info "Starting second stage training with lower learning rate..."
        
        # Create new optimizer with lower learning rate
        second_stage_optimizer = OptimizationOptimisers.Adam(3e-4)
        
        # Reset progress bar
        progressbar = ProgressUnknown("Fine-tuning...", showspeed=true)
        
        # Create new problem with previous solution as initial condition
        prob = remake(prob, u0=res.u)
        
        # Train for additional iterations
        res = solve(prob, second_stage_optimizer, maxiters=2000, cb=callback)
        
        @info "Second stage training completed with loss = $(res.objective)"
    end
    
    finish!(progressbar)
    
    return res, discretization, sym_prob
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
    predict_u::Function
    predict_λ::Function
    ts::Vector{Float64}
    xs::Vector{Float64}
    us::Vector{Vector{Float64}}
    λs::Vector{Vector{Float64}}
    final_loss::Float64
    pde_losses::Vector{Float64}
    bc_losses::Vector{Float64}
end

"""
    train_and_predict(pdesystem, model_config::ModelConfig, training_config::TrainingConfig)

Train a model and generate predictions.
"""
function train_and_predict(pdesystem, model_config::ModelConfig, training_config::TrainingConfig)
    # Create neural networks (one for u and one for λ)
    chains = create_neural_network(model_config, 2, 1)  # 2 inputs (t, x), 1 output per network
    
    # Train model
    res, discretization, sym_prob = train_model(pdesystem, chains, training_config)
    
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
        predict_u,
        predict_λ,
        ts,
        xs,
        us,
        λs,
        res.objective,
        pde_losses,
        bc_losses
    )
end 