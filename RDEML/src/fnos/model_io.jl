import DrWatson: _wsave, wload
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
        "learning_rates" => config.history.learning_rates,
        "epochs" => config.history.epochs,
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
    config =  dict["full_config"]    
    @assert config isa FNOConfig "The full config is not an FNOConfig"
    return config
end
DrWatson.default_allowed(::FNOConfig) = (NTuple, Int, Function)

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
function DrWatson.wload(filename::AbstractString, ::Type{FNOConfig})
    # Load the dictionary from file
    dict = DrWatson.wload(filename)
    # Convert to FNOConfig
    return dict_to_fnoconfig(dict)
end


