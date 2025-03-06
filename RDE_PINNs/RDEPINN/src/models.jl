"""
    ModelConfig

Configuration for neural network models used in Rotating Detonation Engine (RDE) simulations.
"""
struct ModelConfig
    hidden_sizes::Vector{Int}
    activation::Function
    init_strategy::Symbol
end

"""
    default_model_config(; hidden_sizes=[32, 32, 32], activation=tanh, init_strategy=:glorot_uniform)

Create a default model configuration for RDE PINN models.
"""
function default_model_config(; hidden_sizes=[32, 32, 32], activation=tanh, init_strategy=:glorot_uniform)
    return ModelConfig(hidden_sizes, activation, init_strategy)
end

"""
    Base.show(io::IO, config::ModelConfig)

Custom display for ModelConfig objects.
"""
function Base.show(io::IO, config::ModelConfig)
    println(io, "ModelConfig:")
    println(io, "├─ Hidden Sizes: $(config.hidden_sizes)")
    println(io, "├─ Activation: $(config.activation)")
    println(io, "└─ Init Strategy: $(config.init_strategy)")
end

"""
    Base.show(io::IO, ::MIME"text/plain", config::ModelConfig)

Detailed display for ModelConfig objects.
"""
function Base.show(io::IO, ::MIME"text/plain", config::ModelConfig)
    println(io, "ModelConfig:")
    println(io, "├─ Architecture: $(length(config.hidden_sizes)) hidden layers")
    println(io, "├─ Hidden Sizes: $(config.hidden_sizes)")
    println(io, "├─ Total Parameters: $(sum(config.hidden_sizes) + 2*sum(config.hidden_sizes[1:end-1]) + config.hidden_sizes[end])")
    println(io, "├─ Activation: $(config.activation)")
    println(io, "└─ Init Strategy: $(config.init_strategy)")
end

"""
    small_model_config()

Create a small model configuration for quick testing of RDE PINNs.
"""
function small_model_config()
    return ModelConfig([16, 16], tanh, :glorot_uniform)
end

"""
    medium_model_config()

Create a medium-sized model configuration for RDE PINNs.
"""
function medium_model_config()
    return ModelConfig([32, 32, 32], tanh, :glorot_uniform)
end

"""
    large_model_config()

Create a large model configuration for more complex RDE simulations.
"""
function large_model_config()
    return ModelConfig([64, 64, 64, 64], tanh, :glorot_uniform)
end

"""
    deep_model_config()

Create a deep model configuration for more complex RDE simulations.
"""
function deep_model_config()
    return ModelConfig([32, 32, 32, 32, 32], tanh, :glorot_uniform)
end

"""
    tanh_model_config()

Create a model configuration with tanh activation for RDE PINNs.
"""
function tanh_model_config()
    return ModelConfig([32, 32, 32], tanh, :glorot_uniform)
end

"""
    create_neural_network(config::ModelConfig, input_dim::Int, output_dim::Int)

Create neural networks based on the configuration for solving RDE problems.
Uses Lux.jl for neural network creation.

Returns a list of neural networks, one for each output variable (u and λ).
"""
function create_neural_network(config::ModelConfig, input_dim::Int, output_dim::Int)
    # For RDE problems, we need two networks (one for u and one for λ)
    chains = []
    
    # Create two identical networks
    for _ in 1:2
        layers = []
        
        # Input layer to first hidden layer
        push!(layers, Lux.Dense(input_dim => config.hidden_sizes[1], config.activation; init_weight=get_init_strategy(config.init_strategy)))
        
        # Hidden layers
        for i in 1:(length(config.hidden_sizes) - 1)
            push!(layers, Lux.Dense(config.hidden_sizes[i] => config.hidden_sizes[i+1], config.activation; init_weight=get_init_strategy(config.init_strategy)))
        end
        
        # Output layer
        push!(layers, Lux.Dense(config.hidden_sizes[end] => output_dim; init_weight=get_init_strategy(config.init_strategy)))
        
        # Add the chain to our list
        push!(chains, Lux.Chain(layers...))
    end
    
    return chains
end

"""
    get_init_strategy(strategy::Symbol)

Get the initialization strategy for neural network weights in RDE models.
Uses Lux.jl initializers.
"""
function get_init_strategy(strategy::Symbol)
    if strategy == :glorot_uniform
        return Lux.glorot_uniform
    elseif strategy == :glorot_normal
        return Lux.glorot_normal
    elseif strategy == :kaiming_uniform
        return Lux.kaiming_uniform
    elseif strategy == :kaiming_normal
        return Lux.kaiming_normal
    else
        @warn "Unknown initialization strategy: $strategy. Using glorot_uniform."
        return Lux.glorot_uniform
    end
end 