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
end

# =============================================================================
# HYPERPARAMETER SAMPLING
# =============================================================================

function sample_hyperparams(rng::AbstractRNG)
    return Dict(
        "gamma" => rand(rng, 0.92f0 .. 0.999f0),  # 0.85-0.99
        "gae_lambda" => rand(rng, 0.8f0 .. 0.98f0),  # 0.9-1.0
        "clip_range" => rand(rng, 0.1f0 .. 0.3f0),  # 0.1-0.3
        "vf_coef" => rand(rng, 0.2f0 .. 0.8f0),  # 0.1-1.0
        "ent_coef" => rand(rng, [0f0 , 0.01f0]),  # 0.0-0.01
        "learning_rate" => 10^(rand(rng, -4.0f0 .. -3.5f0)),  # 
        "batch_size" => rand(rng, [32, 64, 128]),
        "n_steps" => rand(rng, [64, 128, 256, 512, 1024]),
        "epochs" => rand(rng, [5, 10, 15, 20]),
        "n_envs" => rand(rng, [2, 4, 8, 16]),
        "normalizeWrapper" => rand(rng, [true, false]),
        "scalingWrapper" => rand(rng, [true, false])
    )
end

# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

function evaluate_trained_agent(agent, env, n_episodes::Int=100)
    set_training!(env, false)
    eval_stats = evaluate_agent(agent, env; n_eval_episodes=n_episodes)
    set_training!(env, true)
    return eval_stats.mean_reward
end

function run_single_trial(params::Dict, experiment_name::String)
    # Create algorithm
    alg = PPO(
        gamma=Float32(params["gamma"]),
        gae_lambda=Float32(params["gae_lambda"]),
        clip_range=Float32(params["clip_range"]),
        vf_coef=Float32(params["vf_coef"]),
        ent_coef=Float32(params["ent_coef"])
    )
    # Create environment
    if params["scalingWrapper"]
        envs = [PendulumEnv() |> ScalingWrapperEnv for _ in 1:params["n_envs"]]
    else
        envs = [PendulumEnv() for _ in 1:params["n_envs"]]
    end
    env = MultiThreadedParallelEnv(envs)
    env = MonitorWrapperEnv(env, 50)
    if params["normalizeWrapper"]
        env = NormalizeWrapperEnv(env, gamma=alg.gamma)
    end

    # Create agent
    policy = ActorCriticPolicy(observation_space(env), action_space(env))
    agent = ActorCriticAgent(
        policy;
        learning_rate=Float32(params["learning_rate"]),
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        verbose=0,
        rng=Xoshiro(params["seed"]),
        log_dir=datadir("experiments", experiment_name, "logs", "trial_$(params["trial_id"])_seed_$(params["seed_idx"])")
    )

    
    DRiL.TensorBoardLogger.write_hparams!(agent.logger, params, ["env/ep_rew_mean", "env/ep_len_mean", "train/loss"])
    # Train
    t_start = time()
    learn!(agent, env, alg; max_steps=Int(params["max_steps_per_trial"]))
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
        params = sample_hyperparams(rng)
        params["trial_id"] = trial
        params["max_steps_per_trial"] = Int(config.max_steps_per_trial)

        # Run multiple seeds for this configuration
        for seed_idx in 1:config.n_seeds
            seed_params = copy(params)
            seed_params["seed"] = config.random_seed + trial * 1000 + seed_idx
            seed_params["seed_idx"] = seed_idx

            # Use DrWatson's produce_or_load
            filename = "trial_$(trial)_seed_$(seed_idx)"
            result = produce_or_load(
                seed_params,
                datadir("experiments", config.experiment_name, "results");
                filename) do params
                return run_single_trial(params, config.experiment_name)
            end
        end
    end

    return nothing
end

# =============================================================================
# ANALYSIS
# =============================================================================

function analyze_results(experiment_name::String)
    # Collect all results
    df = collect_results(datadir("experiments", experiment_name, "results"))

    # Group by trial_id and aggregate across seeds
    grouped_df = combine(groupby(df, :trial_id)) do group
        (
            # Hyperparameters (same for all seeds)
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

            # Aggregated results
            mean_return=mean(group.eval_return),
            std_return=std(group.eval_return),
            n_seeds=nrow(group)
        )
    end

    # Find best configuration
    best_idx = argmax(grouped_df.mean_return)
    best_config = grouped_df[best_idx, :]

    # Save results
    experiment_dir = datadir("experiments", experiment_name)
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

function get_best_hyperparams(experiment_name::String)
    _, best_config = analyze_results(experiment_name)

    params = Dict{String,Any}()
    for param in [:gamma, :gae_lambda, :clip_range, :vf_coef, :ent_coef, :learning_rate, :batch_size, :n_steps, :epochs, :n_envs]
        params[string(param)] = best_config[param]
    end

    return params
end

# =============================================================================
# MAIN FUNCTION
# =============================================================================

function main(; n_trials::Int=50, max_steps_per_trial::Int=30_000, experiment_name::String="ppo_search_$(Dates.format(now(), "yyyy-mm-dd_HH-MM"))")
    config = SearchConfig(
        n_trials=n_trials,
        max_steps_per_trial=max_steps_per_trial,
        experiment_name=experiment_name
    )

    # Run search
    run_hyperparameter_search(config)

    # Analyze results
    grouped_df, best_config = analyze_results(experiment_name)

    @info "Hyperparameter search complete!"
    @info "Results saved to: $(datadir("experiments", experiment_name))"

    return grouped_df, best_config
end

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

# Quick test: df, best_config = main(n_trials=5, max_steps_per_trial=5_000)
# Full search: df, best_config = main(n_trials=256, max_steps_per_trial=100_000)
# Get best params: get_best_hyperparams("your_experiment_name")