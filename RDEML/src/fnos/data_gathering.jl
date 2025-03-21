struct DataGatherer
    policies::Vector{Policy}
    reset_strategies::Vector{AbstractReset}
    n_runs::Int
end

total_runs(data_gatherer::DataGatherer) = length(data_gatherer.policies)*length(data_gatherer.reset_strategies)*data_gatherer.n_runs

function Base.show(io::IO, gatherer::DataGatherer)
    if get(io, :compact, false)::Bool
        print(io, "DataGatherer($(length(gatherer.policies))×$(length(gatherer.reset_strategies))×$(gatherer.n_runs))")
    else
        print(io, "DataGatherer($(length(gatherer.policies)) policies, $(length(gatherer.envs)) envs, ",
              "$(length(gatherer.reset_strategies)) reset strategies, $(gatherer.n_runs) runs per config)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", gatherer::DataGatherer)
    total = total_runs(gatherer)
    
    # Get unique policy types
    policy_types = unique(typeof.(gatherer.policies))
    
    # Get unique reset strategy types
    reset_strategy_types = unique(typeof.(gatherer.reset_strategies))
    
    # Main header
    println(io, "DataGatherer:")
    println(io, "├─ Policies: $(length(gatherer.policies)) ($(length(policy_types)) types)")
    
    # Show policy types
    for (i, policy_type) in enumerate(policy_types)
        count = Base.count(p -> typeof(p) == policy_type, gatherer.policies)
        is_last = i == length(policy_types)
        prefix = is_last ? "└─ " : "├─ "
        println(io, "│  $(prefix)$(policy_type): $(count)")
    end
    
    println(io, "├─ Reset Strategies: $(length(gatherer.reset_strategies)) ($(length(reset_strategy_types)) types)")
    
    # Show reset strategy types
    for (i, reset_type) in enumerate(reset_strategy_types)
        count = Base.count(r -> typeof(r) == reset_type, gatherer.reset_strategies)
        is_last = i == length(reset_strategy_types)
        prefix = is_last ? "└─ " : "├─ "
        println(io, "│  $(prefix)$(reset_type): $(count)")
    end
    
    println(io, "├─ Runs per config: $(gatherer.n_runs)")
    println(io, "└─ Total runs: $(total)")
end

struct DataSetInfo
    sim_data::PolicyRunData
    policy::Policy
    reset_strategy::AbstractReset
    run::Int
    steps::Int
end

function Base.show(io::IO, info::DataSetInfo)
    if get(io, :compact, false)::Bool
        print(io, "DataSetInfo($(typeof(info.policy).name.name), $(typeof(info.reset_strategy).name.name), run $(info.run))")
    else
        print(io, "DataSetInfo(policy: $(typeof(info.policy)), reset: $(typeof(info.reset_strategy)), ",
              "run: $(info.run), steps: $(info.steps))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", info::DataSetInfo)
    println(io, "DataSetInfo:")
    println(io, "├─ Policy: $(info.policy)")

    println(io, "├─ Reset Strategy: $(info.reset_strategy)")
    println(io, "├─ Run: $(info.run)")
    println(io, "├─ Steps: $(info.steps)")
    
    # Calculate simulation duration if available
    duration = info.sim_data.state_ts[end]
    println(io, "└─ Simulation Duration: $(duration)")

end

DrWatson.default_allowed(::DataSetInfo) = (Policy, AbstractReset)
DrWatson.allaccess(::DataSetInfo) = (:policy, :reset_strategy)

function DrWatson._wsave(filename::AbstractString, config::DataSetInfo, args...; kwargs...)
    # Convert FNOConfig to a dictionary
    dict = Dict("policy" => config.policy, 
        "reset_strategy" => config.reset_strategy, 
        "run" => config.run, 
        "steps" => config.steps,
        "sim_data" => config.sim_data)
    # Add timestamp
    dict["timestamp"] = Dates.now()
    # Pass to the standard save method for dictionaries
    DrWatson._wsave(filename, dict, args...; kwargs...)
end

function DrWatson.wload(filename::AbstractString, ::Type{DataSetInfo})
    dict = DrWatson.wload(filename)
    return DataSetInfo(dict["sim_data"], dict["policy"], dict["reset_strategy"], dict["run"], dict["steps"])
end

make_env(target_shock_count=1) = RDEEnv(RDEParam(tmax = 400.0f0),
    dt = 1.0f0,
    observation_strategy=SectionedStateObservation(minisections=32, target_shock_count=target_shock_count),
    reset_strategy=ShiftReset(RandomShockOrCombination())
)

function get_data_policies()
    env = make_env()
    policies = [ConstantRDEPolicy(env),
    ScaledPolicy(SinusoidalRDEPolicy(env, w_1=0f0, w_2=0.1f0), 0.05f0),
    ScaledPolicy(SinusoidalRDEPolicy(env, w_1=0f0, w_2=0.05f0), 0.01f0),
    ScaledPolicy(RandomRDEPolicy(env), 0.2f0),
    ScaledPolicy(RandomRDEPolicy(env), 0.1f0),
    ScaledPolicy(RandomRDEPolicy(env), 0.05f0),
    StepwiseRDEPolicy(env, [20.0f0, 100.0f0, 200.0f0, 350.0f0], 
    [0.64f0, 0.86f0, 0.64f0, 0.96f0])
    ]
    for i in 1:4
        push!(policies, load_best_policy("transition_rl_9", project_path=joinpath(homedir(), "Code", "DRL_RDE"),
        filter_fn=df -> df.target_shock_count == i, name="t9-target-$(i)-best", target_shock_count=i)[1])
    end
    return policies
end

function get_data_reset_strategies()
    return [ShiftReset(NShock(1)),
    ShiftReset(NShock(2)),
    ShiftReset(NShock(3)),
    ShiftReset(NShock(4)),
    ShiftReset(RandomCombination()),
    ShiftReset(RandomCombination()),
    ShiftReset(SineCombination()),
    ShiftReset(SineCombination())
    ]
end

function setup_env!(env, policy, reset_strategy)
    env.prob.reset_strategy = reset_strategy
    target_shock_count = hasfield(typeof(policy), :target_shock_count) ? 
                         getfield(policy, :target_shock_count) : 
                         1
    env.observation_strategy= SectionedStateObservation(target_shock_count=target_shock_count)
end

function generate_data(data_gatherer::DataGatherer; dataset_name="datasets")
    tot_runs = total_runs(data_gatherer)
    prog = Progress(tot_runs, "Collecting data...")
    for policy in data_gatherer.policies
        for reset_strategy in data_gatherer.reset_strategies
            for run_i in 1:data_gatherer.n_runs
                env = get_env(policy)
                if isnothing(env)
                    env = make_env()
                end
                setup_env!(env, policy, reset_strategy)
                sim_data = run_policy(policy, env)

                data_set_info = DataSetInfo(sim_data, policy, reset_strategy, run_i, length(sim_data.states))
                safesave(datadir(dataset_name, savename(data_set_info, "jld2")), data_set_info)

                if env.terminated
                    @info "Policy $(policy) with reset strategy
                         $(reset_strategy) terminated at run $run_i"
                end
                
                next!(prog, showvalues=[("$(policy), $(reset_strategy)",run_i)])
            end
        end
    end
    @info "Done collecting data, collected $(length(data_gatherer.policies)) policies 
        × $(length(data_gatherer.reset_strategies)) reset strategies × $(data_gatherer.n_runs) 
        runs = $(tot_runs) total runs"
    return nothing
end

function sim_data_to_data_set(sim_data::PolicyRunData)
    n_data = length(sim_data.states)
    N = length(sim_data.states[1]) ÷ 2
    raw_data = zeros(Float32, N, 3, n_data)
    # x_data = @view raw_data[:,:,1:end-1]
    # y_data = @view raw_data[:,1:2,2:end]
    """
    since the recorded injection pressure at index i was the pressure that was in the chamber from 
    t[i-1] to t[i], we need to shift the injection pressure by one time step to the left
    """
    
    for j in eachindex(sim_data.observations)
        obs = sim_data.states[j]
        raw_data[:, 1, j] = obs[1:N]
        raw_data[:, 2, j] = obs[N+1:2N]
        raw_data[:, 3, j] .= sim_data.u_ps[min(j+1, n_data)] #u_p at last time step is repeated, but the last value unused
    end
    return raw_data
end

"""
    DatasetManager{T<:AbstractFloat}

Manages dataset batches with support for shuffling between epochs.

# Fields
- `raw_x_data`: Original unbatched feature data
- `raw_y_data`: Original unbatched target data
- `current_batches`: Current batched data as vector of (x,y) tuples
- `batch_size`: Size of individual batches (if specified)
- `n_batches`: Number of batches (if specified)
- `rng`: Random number generator for shuffling
"""
mutable struct DatasetManager{T<:AbstractFloat}
    raw_x_data::Array{T,3}           # Original unbatched x data
    raw_y_data::Array{T,3}           # Original unbatched y data
    current_batches::Vector{Tuple{Array{T,3}, Array{T,3}}}  # Current batched data
    batch_size::Union{Int, Nothing}
    n_batches::Union{Int, Nothing}
    rng::AbstractRNG
    
    function DatasetManager(raw_x_data::Array{T,3}, raw_y_data::Array{T,3};
                            max_batch_size::Int=2^17,
                            batches::Union{Int,Nothing}=nothing,
                            shuffle::Bool=true,
                            rng=Random.default_rng()) where {T<:AbstractFloat}
        
        # Create initial batches
        manager = new{T}(raw_x_data, raw_y_data, [], 
                        (isnothing(batches) ? max_batch_size : nothing), 
                        batches, rng)
        
        # Generate initial batch arrangement
        shuffle_batches!(manager, shuffle=shuffle)
        
        return manager
    end
end

function Base.show(io::IO, dm::DatasetManager)
    if get(io, :compact, false)::Bool
        print(io, "DatasetManager($(length(dm.current_batches)) batches)")
    else
        print(io, "DatasetManager($(length(dm.current_batches)) batches, $(size(dm.raw_x_data, 3)) samples)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", dm::DatasetManager{T}) where {T}
    println(io, "DatasetManager{$T}:")
    println(io, "  samples: $(size(dm.raw_x_data, 3))")
    println(io, "  batches: $(length(dm.current_batches))")
    println(io, "  features shape: $(size(dm.raw_x_data))")
    println(io, "  targets shape: $(size(dm.raw_y_data))")
    if !isnothing(dm.batch_size)
        println(io, "  batch size: $(dm.batch_size)")
    elseif !isnothing(dm.n_batches)
        println(io, "  fixed batch count: $(dm.n_batches)")
    end
end

"""
    shuffle_batches!(dm::DatasetManager; shuffle::Bool=true)

Regenerate batches for the dataset, optionally with shuffling.

# Arguments
- `dm::DatasetManager`: The dataset manager to update
- `shuffle::Bool=true`: Whether to shuffle data before creating batches
- `max_batch_size::Union{Int,Nothing}=nothing`: Optional new max batch size
- `batches::Union{Int,Nothing}=nothing`: Optional new number of batches

# Returns
The updated DatasetManager
"""
function shuffle_batches!(dm::DatasetManager; 
                         shuffle::Bool=true,
                         max_batch_size::Union{Int,Nothing}=nothing,
                         batches::Union{Int,Nothing}=nothing)
    
    # Update batch parameters if provided
    if !isnothing(max_batch_size)
        dm.batch_size = max_batch_size
        dm.n_batches = nothing
    end
    if !isnothing(batches)
        dm.n_batches = batches
        dm.batch_size = nothing
    end
    
    # Get data dimensions
    n_samples = size(dm.raw_x_data, 3)
    
    # Create shuffled indices if requested
    if shuffle
        indices = randperm(dm.rng, n_samples)
        x_data = dm.raw_x_data[:,:,indices]
        y_data = dm.raw_y_data[:,:,indices]
    else
        x_data = dm.raw_x_data
        y_data = dm.raw_y_data
    end
    
    # Determine number of batches
    if isnothing(dm.n_batches)
        if n_samples <= dm.batch_size
            n_batches = 1
        else
            n_batches = ceil(Int, n_samples / dm.batch_size)
        end
    else
        n_batches = dm.n_batches
    end
    
    # Calculate batch sizes (as even as possible)
    batch_sizes = fill(div(n_samples, n_batches), n_batches)
    remainder = n_samples % n_batches
    for i in 1:remainder
        batch_sizes[i] += 1
    end
    
    # Create batches
    result = Vector{Tuple{Array{eltype(x_data),3}, Array{eltype(y_data),3}}}(undef, n_batches)
    start_idx = 1
    for i in 1:n_batches
        end_idx = start_idx + batch_sizes[i] - 1
        result[i] = (x_data[:,:,start_idx:end_idx], y_data[:,:,start_idx:end_idx])
        start_idx = end_idx + 1
    end
    
    # Update batches in the manager
    dm.current_batches = result
    
    return dm
end

"""
    prepare_dataset(dataset="datasets"; 
                   max_batch_size::Int=2^17, 
                   batches::Union{Int,Nothing}=nothing, 
                   shuffle::Bool=true, 
                   rng=Random.default_rng())

Prepare a dataset for FNO training by loading data from simulation results and creating a DatasetManager.

# Arguments
- `dataset`: Directory name containing simulation results (relative to project data directory)
- `max_batch_size`: Maximum number of samples per batch (used if `batches` is not specified)
- `batches`: Optional specific number of batches to create (overrides max_batch_size if provided)
- `shuffle`: Whether to randomly shuffle the data before batching
- `rng`: Random number generator for shuffling

# Returns
A DatasetManager that holds the dataset and provides batch management functionality.

# Example
```julia
# Get dataset with automatic batch sizing
data_manager = prepare_dataset("my_simulations", max_batch_size=1000)

# Get dataset with specific number of batches
data_manager = prepare_dataset("my_simulations", batches=5)

# Get dataset without shuffling
data_manager = prepare_dataset("my_simulations", shuffle=false)

# Training with batch shuffling between epochs
for epoch in 1:n_epochs
    for (x, y) in data_manager.current_batches
        # Train on batch
    end
    # Shuffle batches after epoch
    shuffle_batches!(data_manager)
end
```
"""
function prepare_dataset(dataset="datasets"; 
                        max_batch_size::Int=2^17, 
                        batches::Union{Int,Nothing}=nothing, 
                        shuffle::Bool=true, 
                        rng=Random.default_rng())
    df = collect_results(datadir(dataset))
    raw_datas = [sim_data_to_data_set(df.sim_data) for df in eachrow(df)]
    x_datas = [@view raw_data[:,:,1:end-1] for raw_data in raw_datas]
    y_datas = [@view raw_data[:,1:2,2:end] for raw_data in raw_datas]
    combined_x_data = cat(x_datas..., dims=3)
    combined_y_data = cat(y_datas..., dims=3)
    
    # Create dataset manager instead of directly creating batches
    return DatasetManager(
        combined_x_data, 
        combined_y_data;
        max_batch_size=max_batch_size,
        batches=batches,
        shuffle=shuffle,
        rng=rng
    )
end