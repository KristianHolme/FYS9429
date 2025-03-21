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
    target_shock_count = get(policy, :target_shock_count, nothing)
    env.observation_strategy.target_shock_count = target_shock_count
    policy.env = env
end

function generate_data(data_gatherer::DataGatherer; env=make_env(), dataset_name="datasets")
    total_runs = total_runs(data_gatherer)
    prog = Progress(total_runs, "Collecting data...")
    for policy in data_gatherer.policies
        for reset_strategy in data_gatherer.reset_strategies
            for run_i in 1:data_gatherer.n_runs
                setup_env!(env, policy, reset_strategy)
                sim_data = run_policy(policy, env)

                data_set_info = DataSetInfo(sim_data, policy, reset_strategy, run_i, length(sim_data.states))
                safesave(datadir(dataset_name, savename(data_set_info, "jld2")), data_set_info)

                if env.terminated
                    @info "Policy $(policy) with reset strategy
                         $(reset_strategy) terminated at run $run_i"
                end
                
                next!(prog, showvalues=[("$(policy)",run_i)])
            end
        end
    end
    @info "Done collecting data, collected $(length(data_gatherer.policies)) policies 
        × $(length(data_gatherer.reset_strategies)) reset strategies × $(data_gatherer.n_runs) 
        runs = $(total_runs) total runs"
    return run_data, data
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
    prepare_dataset(dataset="datasets"; max_batch_size::Int=2^17, batches::Union{Int,Nothing}=nothing, shuffle::Bool=true, rng=Random.default_rng())

Prepare a dataset for FNO training by loading data from simulation results and splitting into batches.

# Arguments
- `dataset`: Directory name containing simulation results (relative to project data directory)
- `max_batch_size`: Maximum number of samples per batch (used if `batches` is not specified)
- `batches`: Optional specific number of batches to create (overrides max_batch_size if provided)
- `shuffle`: Whether to randomly shuffle the data before batching
- `rng`: Random number generator for shuffling

# Returns
Vector of (x_data, y_data) tuples, where each tuple represents a batch of training data.

# Example
```julia
# Get dataset with automatic batch sizing
batched_data = prepare_dataset("my_simulations", max_batch_size=1000)

# Get dataset with specific number of batches
batched_data = prepare_dataset("my_simulations", batches=5)

# Get dataset without shuffling
batched_data = prepare_dataset("my_simulations", shuffle=false)
```
"""
function prepare_dataset(dataset="datasets"; max_batch_size::Int=2^17, batches::Union{Int,Nothing}=nothing, shuffle::Bool=true, rng=Random.default_rng())
    df = collect_results(datadir(dataset))
    raw_datas = [sim_data_to_data_set(df.sim_data) for df in eachrow(df)]
    x_datas = [@view raw_data[:,:,1:end-1] for raw_data in raw_datas]
    y_datas = [@view raw_data[:,1:2,2:end] for raw_data in raw_datas]
    combined_x_data = cat(x_datas..., dims=3)
    combined_y_data = cat(y_datas..., dims=3)
    
    n_samples = size(combined_x_data, 3)
    
    # Shuffle data if requested
    if shuffle
        perm = randperm(rng, n_samples)
        combined_x_data = combined_x_data[:,:,perm]
        combined_y_data = combined_y_data[:,:,perm]
    end
    
    # Determine number of batches
    if isnothing(batches)
        if n_samples <= max_batch_size
            batches = 1
        else
            batches = ceil(Int, n_samples / max_batch_size)
        end
    end
    
    # Calculate batch sizes (as even as possible)
    batch_sizes = fill(div(n_samples, batches), batches)
    remainder = n_samples % batches
    for i in 1:remainder
        batch_sizes[i] += 1
    end
    
    # Create batches
    result = Vector{Tuple{Array{Float32,3}, Array{Float32,3}}}(undef, batches)
    start_idx = 1
    for i in 1:batches
        end_idx = start_idx + batch_sizes[i] - 1
        result[i] = (combined_x_data[:,:,start_idx:end_idx], combined_y_data[:,:,start_idx:end_idx])
        start_idx = end_idx + 1
    end
    
    return result
end