@kwdef struct TrainingHistory
    losses::Vector{Float32} = []  # Training losses
    test_losses::Vector{Float32} = []  # Test losses (optional)
    epochs::Vector{Int} = []
    learning_rates::Vector{Float32} = []
    training_time::Vector{Float64} = []
end

function Base.show(io::IO, history::TrainingHistory)
    println(io, "TrainingHistory(")
    println(io, "  losses: $(length(history.losses)) values ")
    if !isempty(history.losses)
        println(io, "         min: $(minimum(history.losses))")
        println(io, "         max: $(maximum(history.losses))")
        println(io, "         mean: $(mean(history.losses))")
        println(io, "         latest: $(history.losses[end])")
    end
    if !isempty(history.test_losses)
        println(io, "  test_losses: $(length(history.test_losses)) values ")
        println(io, "         min: $(minimum(history.test_losses))")
        println(io, "         max: $(maximum(history.test_losses))")
        println(io, "         mean: $(mean(history.test_losses))")
        println(io, "         latest: $(history.test_losses[end])")
    end
    println(io, "  epochs=$(history.epochs),")
    println(io, "  learning_rates=$(history.learning_rates)")
    println(io, "  training_time=$(history.training_time)")
    print(io, ")")
end

mutable struct FNOConfig
    chs::NTuple{N,Int} where N
    modes::Int
    activation::Function
    ps::NamedTuple
    st::NamedTuple
    history::TrainingHistory
    function FNOConfig(;chs=(3, 64, 64, 64, 2), modes=16, activation=gelu, rng=Random.default_rng())
        ps, st = Lux.setup(rng, FNO(;chs, modes, activation))
        return new(chs, modes, activation, ps, st, TrainingHistory())
    end
end

Base.show(io::IO, config::FNOConfig) = print(io, "FNOConfig(chs=$(config.chs), modes=$(config.modes), activation=$(config.activation))")

function FNO(config::FNOConfig)
    return FourierNeuralOperator(config.activation; chs=config.chs, modes=(config.modes,), permuted=Val(true))
end
function FNO(;chs=(3, 64, 64, 64, 2), modes=16, activation=gelu)
    return FourierNeuralOperator(activation; chs=chs, modes=(modes,), permuted=Val(true))
end

"""
    evaluate_test_loss(model, ps, st, test_dataloader, dev)

Calculate the mean test loss over all batches in `test_dataloader`.
"""
function evaluate_test_loss(model, ps, st, test_dataloader, dev)
    test_losses = Float32[]
    cpu = cpu_device()
    for (x_test, y_test) in test_dataloader

        y_pred, _ = Lux.apply(model, x_test |> dev, ps, st) |> cpu
        test_loss = MSELoss()(y_pred, y_test)
        push!(test_losses, test_loss)
    end
    return mean(test_losses)
end

function get_init_trainstate(model, ps, st, lr=0.01)
    return Training.TrainState(model, ps, st, OptimizationOptimisers.Adam(lr))
end

function get_init_trainstate(config::FNOConfig, lr=0.01)
    model = FNO(config)
    return Training.TrainState(model, config.ps, config.st, OptimizationOptimisers.Adam(lr))
end

"""
    train!(model, ps, st, data; lr=3f-4, epochs=10, losses=[], test_data=nothing, test_losses=[], dev=cpu_device(), AD=AutoZygote())

Train a model using the provided data and hyperparameters.

# Arguments
- `model`: The model to train
- `ps`: Model parameters
- `st`: Model state
- `data`: DataLoader containing training data
- `lr`: Learning rate
- `epochs`: Number of epochs to train
- `losses`: Vector to store training losses (will be modified in-place)
- `test_data`: Optional DataLoader containing test data 
- `test_losses`: Vector to store test losses (will be modified in-place)
- `dev`: Device to use for training (CPU or GPU)
- `AD`: Automatic differentiation backend
"""
function train!(model, ps, st, data::DataLoader; 
                lr = 3f-4, 
                epochs=10, 
                losses=[], 
                test_data=nothing, 
                test_losses=[], 
                dev=cpu_device(), 
                AD=AutoZygote(),
                tstate=nothing)
    
    if isnothing(tstate)
        tstate = get_init_trainstate(model, ps, st, lr)
    else
        Lux.Optimisers.adjust!(tstate, lr)
    end
    
    # Configure progress bar
    has_test = !isnothing(test_data)
    p = Progress(epochs*length(data), showspeed=true)
    
    for epoch in 1:epochs
        # Training loop
        for (x, y) in dev(data)
            _, loss, _, tstate = Training.single_train_step!(AD, MSELoss(), (x, y), tstate)
            push!(losses, loss)
            
            # Update progress bar
            next!(p, showvalues=[("Train Loss", loss), ("Epoch", epoch), "Test Loss" => isempty(test_losses) ? NaN : test_losses[end]])
        end
        
        # Evaluate on test set after each epoch if available
        if has_test
            test_loss = evaluate_test_loss(model, tstate.parameters, tstate.states, test_data, dev)
            push!(test_losses, test_loss)
        end
    end
    
    return losses, tstate
end

"""
    train!(config::FNOConfig, data, lr::Number, epochs::Int; kwargs...)

Train an FNO model using the provided data and hyperparameters.
"""
function train!(config::FNOConfig, data::DataLoader, 
                lr::Number,
                epochs::Int;
                test_data=nothing,
                dev=cpu_device(),
                AD=AutoZygote(),
                tstate=nothing
               )
    config.ps = config.ps |> dev
    config.st = config.st |> dev
    model = FNO(config)
    hist = config.history
    
    train_time = @elapsed _, tstate = train!(
        model, config.ps, config.st, data;
        losses=hist.losses, 
        test_data=test_data,
        test_losses=hist.test_losses,
        lr, epochs, dev, AD,
        tstate=tstate)
    push!(hist.learning_rates, lr)
    push!(hist.epochs, epochs)
    push!(hist.training_time, train_time)
    
    return tstate
end

"""
    train!(config::FNOConfig, data, lr::AbstractArray{<:Real}, epochs::AbstractArray{<:Int}; kwargs...)

Train an FNO model using multiple learning rates and epoch counts.
"""
function train!(config::FNOConfig, data::DataLoader,
                lr::AbstractArray{<:Real},
                epochs::AbstractArray{<:Int};
                test_data=nothing,
                dev=cpu_device(),
                AD=AutoZygote()
               )
    @assert length(lr) == length(epochs) "lr and epochs must have the same length"
    config.ps = config.ps |> dev
    config.st = config.st |> dev
    tstate = get_init_trainstate(config, lr=lr[1])
    
    for (lr_i, epochs_i) in zip(lr, epochs)
        train!(config, data, lr_i, epochs_i; test_data=test_data, dev=dev, AD=AD, tstate=tstate)
    end
    
    return tstate
end


