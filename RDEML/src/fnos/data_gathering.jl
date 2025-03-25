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
    FNODataset{T<:AbstractFloat}

A dataset structure for FNO training that manages raw data and provides flexible DataLoader creation.
"""
struct FNODataset{T<:AbstractFloat}
    raw_x_data::Array{T,3}  # Original unbatched x data
    raw_y_data::Array{T,3}  # Original unbatched y data
end

function Base.show(io::IO, ds::FNODataset)
    if get(io, :compact, false)::Bool
        print(io, "FNODataset($(size(ds.raw_x_data, 3)) samples)")
    else
        print(io, "FNODataset($(size(ds.raw_x_data, 3)) samples, $(size(ds.raw_x_data, 1))×$(size(ds.raw_x_data, 2))×$(size(ds.raw_x_data, 3)))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", ds::FNODataset{T}) where {T}
    println(io, "FNODataset{$T}:")
    println(io, "  samples: $(size(ds.raw_x_data, 3))")
    println(io, "  features shape: $(size(ds.raw_x_data))")
    println(io, "  targets shape: $(size(ds.raw_y_data))")
end

"""
    create_dataloader(dataset::FNODataset; batch_size=2048, shuffle=true, parallel=true, rng=Random.default_rng())

Create a DataLoader from the raw dataset with the specified configuration.
"""
function create_dataloader(dataset::FNODataset; 
                         batch_size::Int=2048, 
                         shuffle::Bool=true, 
                         parallel::Bool=true,
                         rng=Random.default_rng())
    return DataLoader((dataset.raw_x_data, dataset.raw_y_data); 
                    batchsize=batch_size, 
                    shuffle=shuffle, 
                    parallel=parallel,
                    rng=rng)
end

function number_of_samples(ds::FNODataset)
    return size(ds.raw_x_data, 3)
end

function calculate_number_of_batches(n_samples::Int, batch_size::Int)
    return div(n_samples, batch_size) + (n_samples % batch_size > 0 ? 1 : 0)
end

function calculate_number_of_batches(ds::FNODataset, batch_size::Int)
    n_samples = number_of_samples(ds)
    return calculate_number_of_batches(n_samples, batch_size)
end

"""
    prepare_dataset(dataset="datasets"; 
                    batch_size=2048, 
                    shuffle=true,
                    parallel=true,
                    rng=Random.default_rng(),
                    create_loader=true,
                    test_split=0.2)

Prepare a dataset for FNO training by loading data from simulation results.

# Arguments
- `dataset`: Directory name containing simulation results (relative to project data directory)
- `batch_size`: Number of samples per batch (used to create a DataLoader if requested)
- `shuffle`: Whether to randomly shuffle the data before batching
- `parallel`: Whether to use parallel data loading
- `rng`: Random number generator for shuffling
- `create_loader`: Whether to create and return a DataLoader (true) or just the FNODataset (false)
- `test_split`: Fraction of data to use for testing (between 0 and 1)

# Returns
If `test_split` is 0, returns a single dataset or dataloader.
If `test_split` is > 0, returns a tuple (train_data, test_data) where each is either an FNODataset 
or a DataLoader depending on the value of `create_loader`.

# Example
```julia
# Get raw datasets with train/test split
train_data, test_data = prepare_dataset("my_simulations", create_loader=false, test_split=0.2)

# Create different loaders with different batch sizes
train_loader = create_dataloader(train_data, batch_size=128)
test_loader = create_dataloader(test_data, batch_size=512)

# Or get loaders directly
train_loader, test_loader = prepare_dataset("my_simulations", batch_size=256, test_split=0.2)
```
"""
function prepare_dataset(dataset="datasets"; 
                        batch_size::Int=2048, 
                        shuffle::Bool=true,
                        parallel::Bool=true,
                        rng=Random.default_rng(),
                        create_loader::Bool=true,
                        test_split::Float64=0.0,
                        test_batch_size::Int=2^12)
    df = collect_results(datadir(dataset))
    raw_datas = [sim_data_to_data_set(df.sim_data) for df in eachrow(df)]
    x_datas = [@view raw_data[:,:,1:end-1] for raw_data in raw_datas]
    y_datas = [@view raw_data[:,1:2,2:end] for raw_data in raw_datas]
    combined_x_data = cat(x_datas..., dims=3)
    combined_y_data = cat(y_datas..., dims=3)
    
    # If no test split is requested, return the entire dataset
    if test_split ≤ 0.0
        dataset = FNODataset(combined_x_data, combined_y_data)
        
        # Return either the raw dataset or a configured DataLoader
        if create_loader
            return create_dataloader(dataset; batch_size, shuffle, parallel, rng)
        else
            return dataset
        end
    end
    
    # Otherwise, split the data into train and test sets
    n_samples = size(combined_x_data, 3)
    n_test = round(Int, test_split * n_samples)
    n_train = n_samples - n_test
    
    # Create shuffled indices for splitting
    indices = shuffle ? randperm(rng, n_samples) : collect(1:n_samples)
    train_indices = indices[1:n_train]
    test_indices = indices[n_train+1:end]
    
    # Create train and test datasets
    train_dataset = FNODataset(
        combined_x_data[:,:,train_indices], 
        combined_y_data[:,:,train_indices]
    )
    
    test_dataset = FNODataset(
        combined_x_data[:,:,test_indices], 
        combined_y_data[:,:,test_indices]
    )
    
    if create_loader
        train_loader = create_dataloader(train_dataset; batch_size, shuffle, parallel, rng)
        # For test data, we typically don't need shuffling
        test_loader = create_dataloader(test_dataset; batch_size=test_batch_size, shuffle=false, parallel, rng)
        return train_loader, test_loader
    else
        return train_dataset, test_dataset
    end
end

# Compatibility aliases for transition period
"""
    shuffle_batches!(dataset::FNODataset; batch_size=nothing, shuffle=true, parallel=true, rng=Random.default_rng())

Create a new DataLoader for the dataset with the specified batch size.
This function provides backward compatibility with the old DatasetManager API.
"""
function shuffle_batches!(dataset::FNODataset; 
                        batch_size::Union{Int,Nothing}=nothing,
                        shuffle::Bool=true,
                        parallel::Bool=true,
                        rng=Random.default_rng())
    if isnothing(batch_size)
        batch_size = 2048  # Default batch size
    end
    return create_dataloader(dataset; batch_size, shuffle, parallel, rng)
end

# For backward compatibility - returns the number of batches that would be created with the given batch size
function number_of_batches(ds::FNODataset, batch_size::Int)
    return calculate_number_of_batches(ds, batch_size)
end