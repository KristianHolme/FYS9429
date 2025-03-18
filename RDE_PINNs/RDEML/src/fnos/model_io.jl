using DrWatson
using JLD2
using Dates
using Random
using Printf

# Override DrWatson's _wsave method for FNOConfig type
function DrWatson._wsave(filename::AbstractString, config::FNOConfig, args...; kwargs...)
    # Convert FNOConfig to a dictionary
    dict = fnoconfig_to_dict(config)
    # Add timestamp
    dict["timestamp"] = Dates.now()
    # Pass to the standard save method for dictionaries
    DrWatson._wsave(filename, dict, args...; kwargs...)
end

# Define custom loading function for directly loaded dictionaries
function DrWatson._wload(filename::AbstractString, ::Type{FNOConfig})
    # Load the dictionary from file
    dict = DrWatson._wload(filename)
    # Convert to FNOConfig
    return dict_to_fnoconfig(dict)
end

"""
    save_fno_config_direct(config::FNOConfig, filename::String; kwargs...)

Save the entire FNOConfig structure directly using struct2dict.
This is an alternative approach that simply serializes the entire structure.

# Arguments
- `config::FNOConfig`: The FNO configuration to save
- `filename::String`: Path to save the file
- `kwargs...`: Additional arguments to pass to tagsave

# Returns
- `filename`: The path to the saved file
"""
function save_fno_config_direct(config::FNOConfig, filename::String; safe=true, kwargs...)
    # Ensure the directory exists
    mkpath(dirname(filename))
    
    # Use DrWatson's struct2dict to convert the entire structure
    # Note: This may have limitations with functions and complex structures
    dict = Dict{String, Any}()
    
    # Add some metadata
    dict["timestamp"] = Dates.now()
    dict["fnoconfig"] = config  # Store the entire config
    
    # Latest loss
    if !isempty(config.history.losses)
        dict["latest_loss"] = config.history.losses[end]
    end
    
    # Save the file
    if safe
        tagsave(filename, dict; safe=true, kwargs...)
    else
        tagsave(filename, dict; kwargs...)
    end
    
    return filename
end

"""
    load_fno_config_direct(filename::String)

Load an FNOConfig from a file saved with save_fno_config_direct.

# Returns
- `FNOConfig`: The loaded configuration
"""
function load_fno_config_direct(filename::String)
    # Load the dictionary
    dict = wload(filename)
    
    # If it contains the whole FNOConfig, return it
    if haskey(dict, "fnoconfig") && dict["fnoconfig"] isa FNOConfig
        return dict["fnoconfig"]
    else
        # Fall back to regular loading
        return dict_to_fnoconfig(dict)
    end
end

"""
    save_fno_config(config::FNOConfig, name::String; path=datadir("fno_models"), kwargs...)

Save an FNOConfig to a file using DrWatson.

# Arguments
- `config::FNOConfig`: The FNO configuration to save
- `name::String`: Descriptive name for the model
- `path::String`: Directory to save the model to (default: `datadir("fno_models")`)

# Keywords
- `safe::Bool=true`: Use safesave to avoid overwriting existing files
- Additional keywords are passed to tagsave

# Returns
- Path to the saved file
"""
function save_fno_config(config::FNOConfig, name::String; 
                         path=datadir("fno_models"), 
                         safe=true, 
                         kwargs...)
    # Create directory if it doesn't exist
    mkpath(path)
    
    # Add metadata to config dictionary
    dict = fnoconfig_to_dict(config)
    dict["model_name"] = name
    dict["description"] = get(kwargs, :description, "FNO model saved with DrWatson")
    
    # Remove any keywords that would conflict with tagsave
    kwargs_filtered = filter(p -> p.first ∉ [:model_name, :timestamp, :description], kwargs)
    
    # Use custom savename function to generate the filename
    # This will include model parameters in the filename
    filename = joinpath(path, DrWatson.savename(config, name))
    
    # Save with git information
    if safe
        tagsave(filename, dict; safe=true, kwargs_filtered...)
    else
        tagsave(filename, dict; kwargs_filtered...)
    end
    
    return filename
end

"""
    load_fno_config(filename::String)

Load an FNOConfig from a file.

# Returns
- `FNOConfig`: The loaded configuration
"""
function load_fno_config(filename::String)
    # Use wload to load, which will use our custom _wload method if available
    return wload(filename, FNOConfig)
end

"""
    list_saved_models(; path=datadir("fno_models"), pattern="*.jld2")

List all saved FNO models in the given directory.

# Returns
- Vector of filenames
"""
function list_saved_models(; path=datadir("fno_models"), pattern="*.jld2")
    isdir(path) || return String[]
    return filter(f -> occursin(pattern, f), readdir(path, join=true))
end

"""
    produce_or_load_fno(name::String, params::Dict; 
                       path=datadir("fno_models"), 
                       suffix="jld2", 
                       force=false,
                       verbose=true)

A wrapper around DrWatson's produce_or_load for FNO models.
If a model with the given parameters exists, load it, otherwise create and train it.

# Arguments
- `name::String`: Base name for the model
- `params::Dict`: Parameters for the model
- `train_function`: Function to call if model needs to be created (fn(params) -> FNOConfig)

# Returns
- `FNOConfig`: The loaded or newly created configuration
- `filename`: Path to the file where the model is saved
"""
function produce_or_load_fno(name::String, params::Dict, train_function::Function; 
                            path=datadir("fno_models"), 
                            suffix="jld2", 
                            force=false,
                            verbose=true)
    
    # Create a produce function that creates and saves an FNO model
    function _produce(params)
        if verbose
            @info "Creating new FNO model with parameters: $(params)"
        end
        
        # Call the provided train_function to create and train the model
        config = train_function(params)
        
        # Convert to dictionary for saving
        result = fnoconfig_to_dict(config)
        
        # Add metadata
        result["model_name"] = name
        result["timestamp"] = Dates.now()
        result["parameters"] = params
        
        return result
    end
    
    # Use DrWatson's produce_or_load
    result_dict, filename = produce_or_load(
        path,
        params;
        filename=name,
        suffix=suffix,
        force=force,
        _produce=_produce
    )
    
    # Convert back to FNOConfig
    return dict_to_fnoconfig(result_dict), filename
end

"""
    save_prediction(config::FNOConfig, inputs, predictions, name::String; 
                   path=datadir("fno_predictions"), kwargs...)

Save model predictions to a file.

# Arguments
- `config::FNOConfig`: The FNO configuration used
- `inputs`: The input data
- `predictions`: The predicted outputs
- `name::String`: Descriptive name for the predictions

# Returns
- Path to the saved file
"""
function save_prediction(config::FNOConfig, inputs, predictions, name::String; 
                        path=datadir("fno_predictions"), 
                        kwargs...)
    
    # Create directory if it doesn't exist
    mkpath(path)
    
    # Create a dictionary for saving
    dict = Dict{String, Any}(
        "inputs" => inputs,
        "predictions" => predictions,
        "model_config" => fnoconfig_to_dict(config),
        "timestamp" => Dates.now()
    )
    
    # Add any additional metadata
    for (k, v) in kwargs
        dict[string(k)] = v
    end
    
    # Create filename
    timestamp = Dates.format(dict["timestamp"], "yyyymmdd_HHMMSS")
    filename = joinpath(path, "$(name)_prediction_$(timestamp).jld2")
    
    # Save with git information
    tagsave(filename, dict; safe=true)
    
    return filename
end

"""
    fnoconfig_to_dict(config::FNOConfig)

Convert an FNOConfig to a dictionary for saving.
"""
function fnoconfig_to_dict(config::FNOConfig)
    # Create a dictionary with all the fields from the FNOConfig
    dict = Dict{String, Any}(
        "chs" => config.chs,
        "modes" => config.modes,
        "activation" => string(config.activation),  # Convert function to string name
        "ps" => config.ps,
        "st" => config.st,
        "history" => config.history,  # Store the entire TrainingHistory struct
        "full_config" => config       # Store the entire FNOConfig for direct access
    )
    
    # Also add the latest loss value if it exists
    if !isempty(config.history.losses)
        dict["latest_loss"] = config.history.losses[end]
    end
    
    return dict
end

"""
    dict_to_fnoconfig(dict::Dict)

Convert a dictionary to an FNOConfig object.
"""
function dict_to_fnoconfig(dict::Dict)
    # If the dictionary contains the full config, use it directly
    if haskey(dict, "full_config") && dict["full_config"] isa FNOConfig
        return dict["full_config"]
    end
    
    # If it contains the legacy fnoconfig field, return it
    if haskey(dict, "fnoconfig") && dict["fnoconfig"] isa FNOConfig
        return dict["fnoconfig"]
    end
    
    # Otherwise, reconstruct from individual fields
    # Handle activation function
    activation_name = dict["activation"]
    activation = if activation_name == "gelu"
        gelu
    elseif activation_name == "relu"
        relu
    elseif activation_name == "tanh"
        tanh
    else
        @warn "Activation function $activation_name not recognized, defaulting to gelu"
        gelu  # Default to gelu if not recognized
    end
    
    # We need to check how the history is stored
    history = if dict["history"] isa TrainingHistory
        # If it's already a TrainingHistory struct, use it directly
        dict["history"]
    else
        # If it's a dictionary, convert it back to TrainingHistory
        hist_dict = dict["history"]
        TrainingHistory(
            losses = hist_dict["losses"],
            epochs = hist_dict["epochs"],
            learning_rates = hist_dict["learning_rates"]
        )
    end
    
    # Create a new FNOConfig with the history parameter
    config = FNOConfig(chs=dict["chs"], modes=dict["modes"], activation=activation)
    
    # Set the parameters and state
    config.ps = dict["ps"]
    config.st = dict["st"]
    config.history = history
    
    return config
end

# Implement DrWatson.savename for FNOConfig to customize filename generation
function DrWatson.savename(config::FNOConfig, suffix::String = "jld2")
    # Extract the key parameters that should be in the filename
    channels = join(config.chs, "-")
    
    # Get the latest loss if available
    loss_str = if !isempty(config.history.losses)
        @sprintf("loss=%.6f", config.history.losses[end])
    else
        "untrained"
    end
    
    # Create a descriptive filename
    base = "fno_chs=$(channels)_modes=$(config.modes)_act=$(config.activation)_$(loss_str)"
    
    # Add suffix if provided
    if !isempty(suffix)
        return base * "." * suffix
    else
        return base
    end
end

# Also implement an extended version that includes custom identifiers
function DrWatson.savename(config::FNOConfig, id::String, suffix::String = "jld2")
    base = DrWatson.savename(config, "")
    
    # Add timestamp for uniqueness
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    
    # Combine all parts
    if !isempty(suffix)
        return "$(id)_$(base)_$(timestamp).$(suffix)"
    else
        return "$(id)_$(base)_$(timestamp)"
    end
end

