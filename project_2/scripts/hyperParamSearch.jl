using DrWatson
@quickactivate :project_2
using DRiL
using Pendulum
using ProgressMeter
using IntervalSets
using Accessors
##
using Distributed
addprocs(4) #16, 8 is too much! for nam-shub-02
@everywhere using DrWatson
@everywhere @quickactivate :project_2
@everywhere begin
    using DRiL
    using Pendulum
    using Accessors
    using IntervalSets
    using Zygote
    using Random
    using JLD2
end

## searching over learning rate, gamma, gae_lambda, vf_coef, batch_size
@everywhere experiment_seed = 349
function load_initial_configs()
    Random.seed!(experiment_seed)
    n_experiments = 128
    gammas = rand((0.8f0) .. (0.99f0), n_experiments)
    gae_lambdas = rand((0.9f0) .. (1.0f0), n_experiments)
    vf_coefs = rand((0.0f0) .. (1.0f0), n_experiments)
    learning_rates = exp.(rand((-4f0) .. (-1f0), n_experiments))
    batch_sizes = rand([16, 32, 64, 128], n_experiments)
    n_stepss = rand([64, 128, 256, 512, 1024], n_experiments)

    configs = Tuple{Dict{String,Any},String}[]

    for i in 1:n_experiments
        learning_rate = learning_rates[i]
        gamma = gammas[i]
        gae_lambda = gae_lambdas[i]
        vf_coef = vf_coefs[i]
        batch_size = batch_sizes[i]
        n_steps = n_stepss[i]
        push!(configs, (Dict(
                "learning_rate" => learning_rate,
                "gamma" => gamma,
                "gae_lambda" => gae_lambda,
                "vf_coef" => vf_coef,
                "batch_size" => batch_size,
                "n_steps" => n_steps,
                "i" => i
            ), "initial"))
    end
    return configs
end

function load_further_configs(n_experiments=256)
    Random.seed!(experiment_seed)
    gammas = rand((0.8f0) .. (0.99f0), n_experiments)
    gae_lambdas = rand((0.9f0) .. (1.0f0), n_experiments)
    vf_coefs = rand((0.0f0) .. (1.0f0), n_experiments)
    learning_rates = exp.(rand((-4f0) .. (-1f0), n_experiments))
    batch_sizes = rand([16, 32, 64, 128], n_experiments)
    n_stepss = rand([8, 16, 32, 64], n_experiments)

    configs = Tuple{Dict{String,Any},String}[]

    for i in 1:n_experiments
        learning_rate = learning_rates[i]
        gamma = gammas[i]
        gae_lambda = gae_lambdas[i]
        vf_coef = vf_coefs[i]
        batch_size = batch_sizes[i]
        n_steps = n_stepss[i]
        push!(configs, (Dict(
                "learning_rate" => learning_rate,
                "gamma" => gamma,
                "gae_lambda" => gae_lambda,
                "vf_coef" => vf_coef,
                "batch_size" => batch_size,
                "n_steps" => n_steps,
                "i" => i
            ), "second"))
    end
    return configs
end

@everywhere function run_experiment(config)
    params, run = config
    folder = "hyper_search" * "_" * run
    name = savename(params)
    env = default_env()
    agent = default_agent(PendulumEnv(); folder, name)
    alg = default_PPO(PendulumEnv())

    rng = Xoshiro(experiment_seed)

    Accessors.@reset agent.rng = rng
    Accessors.@reset alg.gamma = params["gamma"]
    Accessors.@reset alg.gae_lambda = params["gae_lambda"]
    Accessors.@reset alg.vf_coef = params["vf_coef"]
    Accessors.@reset agent.learning_rate = params["learning_rate"]
    Accessors.@reset agent.batch_size = params["batch_size"]
    Accessors.@reset agent.n_steps = params["n_steps"]
    Accessors.@reset agent.verbose = 0
    metrics = ["env/avg_ep_rew", "train/loss"]
    # DRiL.TensorBoardLogger.write_hparams!(agent.logger, agent, metrics)
    # DRiL.TensorBoardLogger.write_hparams!(agent.logger, alg, metrics)
    DRiL.TensorBoardLogger.write_hparams!(agent.logger, alg, agent, metrics)

    learn_stats = learn!(agent, env, alg; max_steps=100_000)

    result = merge(params,
        Dict("learn_stats" => learn_stats,
            "agent" => agent,
            "alg" => alg)
    )
    safesave(datadir(folder, name * ".jld2"), result)
    return result
end

##
configs = load_further_configs()

function run_experiments(configs)
    n_experiments = length(configs)
    @showprogress @distributed (+) for i in 1:n_experiments
        params, run = configs[i]
        folder = "hyper_search" * "_" * run
        name = savename(params)
        produce_or_load(configs[i], datadir(folder, name * ".jld2")) do cfg
            result = run_experiment(cfg)
        end
    end
    @info "Finished running experiments"
    return nothing
end