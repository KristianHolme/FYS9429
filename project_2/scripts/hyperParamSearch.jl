using DrWatson
@quickactivate :project_2
using DRiL
using Zygote
using ClassicControlEnvironments
using ProgressMeter
using Random
using JLD2
using Statistics
using DataFrames
using IntervalSets
using CSV
using Dates
using Base.Threads
# =============================================================================
# SEARCH CONFIGURATION
# =============================================================================

@kwdef struct SearchConfig
    n_trials::Int = 128
    max_steps_per_trial::Int = 100_000
    experiment_name::String = "ppo_search_$(Dates.format(now(), "yyyy-mm-dd_HH-MM"))"
    n_seeds::Int = 5
    random_seed::Int = 42
    environment::Type = PendulumEnv    # Environment type as a type (e.g., PendulumEnv)
    algorithm::Type{<:DRiL.AbstractAlgorithm} = PPO
end

# =============================================================================
# HYPERPARAMETER SAMPLING
# =============================================================================

function sample_hyperparams(::Type{PPO}, rng::AbstractRNG, env_type::Type=PendulumEnv)
    # Sample n_steps and n_envs first
    n_steps = rand(rng, [16, 32, 64, 128])
    n_envs = rand(rng, [4, 8, 16, 32])

    # Calculate total buffer size
    total_buffer_size = n_steps * n_envs

    # Sample batch_size ensuring it's <= total_buffer_size
    possible_batch_sizes = [32, 64, 128, 256]
    valid_batch_sizes = filter(bs -> bs <= total_buffer_size, possible_batch_sizes)

    if isempty(valid_batch_sizes)
        valid_batch_sizes = [16, 32]
        valid_batch_sizes = filter(bs -> bs <= total_buffer_size, valid_batch_sizes)
    end

    if isempty(valid_batch_sizes)
        valid_batch_sizes = [min(16, total_buffer_size)]
    end

    batch_size = rand(rng, valid_batch_sizes)

    params = Dict{String,Any}(
        "algorithm" => "PPO",
        "gamma" => rand(rng, 0.96f0 .. 1f0),
        "gae_lambda" => rand(rng, 0.7f0 .. 0.98f0),
        "clip_range" => rand(rng, 0.12f0 .. 0.35f0),
        "vf_coef" => rand(rng, 0.2f0 .. 0.8f0),
        "ent_coef" => rand(rng, 0f0 .. 0.01f0),
        "learning_rate" => 10^(rand(rng, -5.0f0 .. -3f0)),
        "batch_size" => batch_size,
        "n_steps" => n_steps,
        "epochs" => rand(rng, [10, 20, 30]),
        "n_envs" => n_envs,
        "normalizeWrapper" => rand(rng, [true, false]),
        "scalingWrapper" => rand(rng, [false])
    )

    # Apply env-specific adjustments via multiple dispatch
    adjust_ppo_params!(params, env_type, rng)
    return params
end

# Default: no changes
adjust_ppo_params!(params::Dict, ::Type, rng::AbstractRNG) = params

# Sparse reward envs
function adjust_ppo_params!(params::Dict, ::Type{MountainCarEnv}, rng::AbstractRNG)
    params["gamma"] = rand(rng, 0.98f0 .. 1f0)
    params["gae_lambda"] = rand(rng, 0.85f0 .. 0.98f0)
    params["ent_coef"] = rand(rng, 0.001f0 .. 0.05f0)
    return params
end
function adjust_ppo_params!(params::Dict, ::Type{AcrobotEnv}, rng::AbstractRNG)
    params["gamma"] = rand(rng, 0.98f0 .. 1f0)
    params["gae_lambda"] = rand(rng, 0.85f0 .. 0.98f0)
    params["ent_coef"] = rand(rng, 0.001f0 .. 0.05f0)
    return params
end

# Easy discrete env
function adjust_ppo_params!(params::Dict, ::Type{CartPoleEnv}, rng::AbstractRNG)
    params["ent_coef"] = rand(rng, 0f0 .. 0.005f0)
    return params
end

# Continuous control envs
function adjust_ppo_params!(params::Dict, ::Type{PendulumEnv}, rng::AbstractRNG)
    params["ent_coef"] = rand(rng, 0f0 .. 0.01f0)
    return params
end
function adjust_ppo_params!(params::Dict, ::Type{MountainCarContinuousEnv}, rng::AbstractRNG)
    params["ent_coef"] = rand(rng, 0f0 .. 0.01f0)
    return params
end

function sample_hyperparams(::Type{SAC}, rng::AbstractRNG, env_type::Type=PendulumEnv)
    # SAC-specific sampling (continuous control)
    n_envs = rand(rng, [1, 2, 4, 8])
    params = Dict{String,Any}(
        "algorithm" => "SAC",
        "gamma" => rand(rng, 0.96f0 .. 0.995f0),
        "learning_rate" => 10^(rand(rng, -5.0f0 .. -3f0)),
        "batch_size" => rand(rng, [64, 128, 256, 512]),
        "n_envs" => n_envs,
        "buffer_capacity" => rand(rng, [100_000, 250_000, 500_000, 1_000_000]),
        "start_steps" => rand(rng, [1_000, 2_000, 5_000, 10_000]),
        "tau" => rand(rng, 0.001f0 .. 0.02f0),
        "train_freq" => rand(rng, [n_envs, 2 * n_envs, 4 * n_envs, 8 * n_envs]),
        "gradient_steps" => rand(rng, [1, 2, 4, -1]),
        "target_update_interval" => rand(rng, [1, 5, 10]),
        # Entropy coefficient mode for logging (actual object is built in get_alg)
        "ent_coef_mode" => rand(rng, ["auto", "fixed"]),
        # If fixed mode is sampled, pick a value; otherwise value is ignored
        "ent_coef" => rand(rng, 0.05f0 .. 0.5f0),
        "normalizeWrapper" => rand(rng, [true, false]),
        "scalingWrapper" => rand(rng, [false])
    )
    return params
end

# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

function create_single_environment(env_type::Type, params::Dict)
    """Create environment instance from a type"""
    base_env = env_type()

    # Apply scaling wrapper if requested
    if params["scalingWrapper"]
        base_env = ScalingWrapperEnv(base_env)
    end

    return base_env
end

function create_env(env_type::Type, params::Dict)
    envs = [create_single_environment(env_type, params) for _ in 1:params["n_envs"]]
    env = MultiThreadedParallelEnv(envs)
    env = MonitorWrapperEnv(env, 50)
    if params["normalizeWrapper"]
        env = NormalizeWrapperEnv(env, gamma=Float32(params["gamma"]))
    end
    return env
end

## removed non-dispatch get_alg; replaced by get_alg(::Type{<:AbstractAlgorithm}, params)

function evaluate_trained_agent(agent, env, n_episodes::Int=100)
    env = set_training(env, false)
    eval_stats = evaluate_agent(agent, env; n_eval_episodes=n_episodes)
    env = set_training(env, true)
    return eval_stats.mean_reward
end

function get_alg(::Type{PPO}, params::Dict)
    return PPO(
        gamma=Float32(params["gamma"]),
        gae_lambda=Float32(params["gae_lambda"]),
        clip_range=Float32(params["clip_range"]),
        vf_coef=Float32(params["vf_coef"]),
        ent_coef=Float32(params["ent_coef"]),
        n_steps=Int(params["n_steps"]),
        batch_size=Int(params["batch_size"]),
        epochs=Int(params["epochs"]),
        learning_rate=Float32(params["learning_rate"])
    )
end

function get_alg(::Type{SAC}, params::Dict)
    ent_coef_obj = params["ent_coef_mode"] == "auto" ? AutoEntropyCoefficient() : FixedEntropyCoefficient(Float32(params["ent_coef"]))
    return SAC(
        learning_rate=Float32(params["learning_rate"]),
        buffer_capacity=Int(params["buffer_capacity"]),
        start_steps=Int(params["start_steps"]),
        batch_size=Int(params["batch_size"]),
        tau=Float32(params["tau"]),
        gamma=Float32(params["gamma"]),
        train_freq=Int(params["train_freq"]),
        gradient_steps=Int(params["gradient_steps"]),
        ent_coef=ent_coef_obj,
        target_update_interval=Int(params["target_update_interval"])
    )
end

function get_hparam_metrics(::Type{PPO})
    return ["env/ep_rew_mean", "env/ep_len_mean", "train/loss"]
end
function get_hparam_metrics(::Type{SAC})
    return ["env/ep_rew_mean", "env/ep_len_mean", "train/actor_loss", "train/critic_loss", "train/entropy_loss"]
end

function create_policy(::Type{PPO}, env, params::Dict)
    return ActorCriticPolicy(observation_space(env), action_space(env))
end

function create_policy(::Type{SAC}, env, params::Dict)
    if !(action_space(env) isa Box)
        error("SAC requires a continuous action space. Environment $(params["environment"]) is not supported.")
    end
    return ContinuousActorCriticPolicy(observation_space(env), action_space(env); critic_type=QCritic())
end

function create_agent(::Type{PPO}, policy, alg, params::Dict, experiment_name::String)
    return ActorCriticAgent(
        policy, alg;
        verbose=0,
        rng=Xoshiro(params["seed"]),
        log_dir=datadir("experiments", params["environment"], experiment_name, "logs", "trial_$(params["trial_id"])_seed_$(params["seed_idx"])")
    )
end

function create_agent(::Type{SAC}, policy, alg, params::Dict, experiment_name::String)
    return SACAgent(
        policy, alg;
        verbose=0,
        rng=Xoshiro(params["seed"]),
        log_dir=datadir("experiments", params["environment"], experiment_name, "logs", "trial_$(params["trial_id"])_seed_$(params["seed_idx"])")
    )
end

env_name(env_type::Type) = string(nameof(env_type))

function run_single_trial(alg_type::Type{<:DRiL.AbstractAlgorithm}, env_type::Type, params::Dict, experiment_name::String)
    alg = get_alg(alg_type, params)

    # Create multiple environments for parallel training
    env = create_env(env_type, params)

    # Policy and agent
    policy = create_policy(alg_type, env, params)
    agent = create_agent(alg_type, policy, alg, params, experiment_name)
    if !isnothing(agent.logger)
        DRiL.TensorBoardLogger.write_hparams!(agent.logger, params, get_hparam_metrics(alg_type))
    end

    # Train
    t_start = time()
    learn!(agent, env, alg, Int(params["max_steps_per_trial"]))
    t_end = time()
    elapsed_time = t_end - t_start
    # Evaluate
    t_start = time()
    eval_return = evaluate_trained_agent(agent, env)
    t_end = time()
    eval_time = t_end - t_start

    return merge(params, Dict("eval_return" => eval_return,
        "train_time" => elapsed_time,
        "eval_time" => eval_time))
end

# =============================================================================
# MAIN SEARCH
# =============================================================================

function run_hyperparameter_search(config::SearchConfig=SearchConfig())
    rng = Xoshiro(config.random_seed)

    @showprogress @threads for trial in 1:config.n_trials
        # Sample hyperparameters
        params = sample_hyperparams(config.algorithm, rng, config.environment)
        params["trial_id"] = trial
        params["max_steps_per_trial"] = Int(config.max_steps_per_trial)
        params["environment"] = string(config.environment)
        params["algorithm"] = string(config.algorithm)

        # Run multiple seeds for this configuration
        for seed_idx in 1:config.n_seeds
            seed_params = copy(params)
            seed_params["seed"] = config.random_seed + trial * 1000 + seed_idx
            seed_params["seed_idx"] = seed_idx

            # Use DrWatson's produce_or_load
            filename = "trial_$(trial)_seed_$(seed_idx)"
            result = produce_or_load(
                seed_params,
                datadir("experiments", env_name(config.environment), config.experiment_name, "results");
                filename) do params
                return run_single_trial(config.algorithm, config.environment, params, config.experiment_name)
            end
        end
    end

    return nothing
end

# =============================================================================
# ANALYSIS
# =============================================================================

function analyze_results(::Type{PPO}, experiment_name::String, env_type::Type=PendulumEnv)
    df = collect_results(datadir("experiments", env_name(env_type), experiment_name, "results"))

    grouped_df = combine(groupby(df, :trial_id)) do group
        (
            gamma=first(group.gamma),
            gae_lambda=first(group.gae_lambda),
            clip_range=first(group.clip_range),
            vf_coef=first(group.vf_coef),
            ent_coef=first(group.ent_coef),
            learning_rate=first(group.learning_rate),
            batch_size=first(group.batch_size),
            n_steps=first(group.n_steps),
            epochs=first(group.epochs),
            n_envs=first(group.n_envs),
            normalizeWrapper=first(group.normalizeWrapper),
            scalingWrapper=first(group.scalingWrapper),
            mean_return=mean(group.eval_return),
            std_return=std(group.eval_return),
            n_seeds=nrow(group)
        )
    end

    best_idx = argmax(grouped_df.mean_return)
    best_config = grouped_df[best_idx, :]

    experiment_dir = datadir("experiments", env_name(env_type), experiment_name)
    CSV.write(joinpath(experiment_dir, "individual_runs.csv"), df)
    CSV.write(joinpath(experiment_dir, "aggregated_results.csv"), grouped_df)

    @info "Best configuration:"
    @info "  Mean return: $(round(best_config.mean_return, digits=3)) ± $(round(best_config.std_return, digits=3))"
    @info "  Hyperparameters:"
    for param in [:gamma, :gae_lambda, :clip_range, :vf_coef, :ent_coef, :learning_rate, :batch_size, :n_steps, :epochs]
        @info "    $param: $(best_config[param])"
    end

    return grouped_df, best_config
end

function analyze_results(::Type{SAC}, experiment_name::String, env_type::Type=PendulumEnv)
    df = collect_results(datadir("experiments", env_name(env_type), experiment_name, "results"))

    grouped_df = combine(groupby(df, :trial_id)) do group
        (
            gamma=first(group.gamma),
            learning_rate=first(group.learning_rate),
            batch_size=first(group.batch_size),
            n_envs=first(group.n_envs),
            buffer_capacity=first(group.buffer_capacity),
            start_steps=first(group.start_steps),
            tau=first(group.tau),
            train_freq=first(group.train_freq),
            gradient_steps=first(group.gradient_steps),
            target_update_interval=first(group.target_update_interval),
            ent_coef_mode=first(group.ent_coef_mode),
            ent_coef=first(group.ent_coef),
            normalizeWrapper=first(group.normalizeWrapper),
            scalingWrapper=first(group.scalingWrapper),
            mean_return=mean(group.eval_return),
            std_return=std(group.eval_return),
            n_seeds=nrow(group)
        )
    end

    best_idx = argmax(grouped_df.mean_return)
    best_config = grouped_df[best_idx, :]

    experiment_dir = datadir("experiments", env_name(env_type), experiment_name)
    CSV.write(joinpath(experiment_dir, "individual_runs.csv"), df)
    CSV.write(joinpath(experiment_dir, "aggregated_results.csv"), grouped_df)

    @info "Best configuration:"
    @info "  Mean return: $(round(best_config.mean_return, digits=3)) ± $(round(best_config.std_return, digits=3))"
    @info "  Hyperparameters:"
    for param in [:gamma, :learning_rate, :batch_size, :n_envs, :buffer_capacity, :start_steps, :tau, :train_freq, :gradient_steps, :target_update_interval, :ent_coef_mode, :ent_coef]
        @info "    $param: $(best_config[param])"
    end

    return grouped_df, best_config
end

function analyze_results(experiment_name::String, env_type::Type=PendulumEnv, algorithm::Type{<:DRiL.AbstractAlgorithm}=PPO)
    return analyze_results(algorithm, experiment_name, env_type)
end

function get_best_hyperparams(::Type{PPO}, experiment_name::String, env_type::Type=PendulumEnv)
    _, best_config = analyze_results(PPO, experiment_name, env_type)
    params = Dict{String,Any}()
    for param in [:gamma, :gae_lambda, :clip_range, :vf_coef, :ent_coef, :learning_rate, :batch_size, :n_steps, :epochs, :n_envs, :normalizeWrapper, :scalingWrapper]
        params[string(param)] = best_config[param]
    end
    return params
end

function get_best_hyperparams(::Type{SAC}, experiment_name::String, env_type::Type=PendulumEnv)
    _, best_config = analyze_results(SAC, experiment_name, env_type)
    params = Dict{String,Any}()
    for param in [:gamma, :learning_rate, :batch_size, :n_envs, :buffer_capacity, :start_steps, :tau, :train_freq, :gradient_steps, :target_update_interval, :ent_coef_mode, :ent_coef, :normalizeWrapper, :scalingWrapper]
        params[string(param)] = best_config[param]
    end
    return params
end

function get_best_hyperparams(experiment_name::String, env_type::Type=PendulumEnv, algorithm::Type{<:DRiL.AbstractAlgorithm}=PPO)
    return get_best_hyperparams(algorithm, experiment_name, env_type)
end

# =============================================================================
# MAIN FUNCTION
# =============================================================================

function main(; n_trials::Int=50, max_steps_per_trial::Int=30_000, experiment_name::String="ppo_search_$(Dates.format(now(), "yyyy-mm-dd_HH-MM"))", environment::Type=PendulumEnv, algorithm::Type{<:DRiL.AbstractAlgorithm}=PPO)
    config = SearchConfig(
        n_trials=n_trials,
        max_steps_per_trial=max_steps_per_trial,
        experiment_name=experiment_name,
        environment=environment,
        algorithm=algorithm
    )

    # Run search
    run_hyperparameter_search(config)


    # Analyze results
    grouped_df, best_config = analyze_results(experiment_name, environment, algorithm)

    @info "Hyperparameter search complete!"
    @info "Results saved to: $(datadir("experiments", env_name(environment), experiment_name))"

    return grouped_df, best_config
end

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

# List all available environments
# print_environment_info()

# Single environment searches (PPO by default):
# df, best_config = main(n_trials=5, max_steps_per_trial=5_000, environment=PendulumEnv)
# df, best_config = main(n_trials=5, max_steps_per_trial=5_000, environment=MountainCarEnv)
# df, best_config = main(n_trials=5, max_steps_per_trial=10_000, environment=AcrobotEnv)
# df, best_config = main(n_trials=5, max_steps_per_trial=5_000, environment=CartPoleEnv)
# df, best_config = main(n_trials=5, max_steps_per_trial=10_000, environment=MountainCarContinuousEnv)

# SAC examples (requires continuous actions):
# df, best_config = main(n_trials=5, max_steps_per_trial=30_000, environment=PendulumEnv, algorithm=SAC)
# df, best_config = main(n_trials=50, max_steps_per_trial=50_000, environment=MountainCarContinuousEnv, algorithm=SAC)

# Full hyperparameter search for production:
# df, best_config = main(n_trials=256, max_steps_per_trial=100_000, environment=MountainCarEnv)

# Get best hyperparameters from completed search:
# Get best hyperparameters from completed search:
# best_params_ppo = get_best_hyperparams("your_experiment_name", MountainCarEnv, PPO)
# best_params_sac = get_best_hyperparams("your_experiment_name", MountainCarContinuousEnv, SAC)

# Analyze existing results:
# grouped_df, best_config = analyze_results("your_experiment_name", MountainCarEnv, PPO)
# grouped_df, best_config = analyze_results("your_experiment_name", MountainCarContinuousEnv, SAC)