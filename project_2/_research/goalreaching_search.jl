using DrWatson
@quickactivate :project_2
using Lux
using DRiL
using Zygote
using CairoMakie
using Accessors
using IntervalSets
using Random
using Base.Threads
using ProgressMeter
##
function sample_goal_params()
    alg_params = (
        gae_lambda=rand(0.85f0 .. 0.95f0),
        ent_coef=rand(0.00f0 .. 0.01f0),
        vf_coef=rand(0.1f0 .. 0.5f0),
        gamma=rand(0.9f0 .. 0.99f0),
    )
    agent_params = (
        learning_rate=10^(rand(-5f0 .. -1f0)),
        batch_size=rand([16, 32, 64]),
        n_steps=rand([128, 256, 512]),
        epochs=rand([5, 10, 15]),
    )
    return alg_params, agent_params
end

function run_experiment(n_configs=10, n_seeds=5, run=1)
    p = Progress(n_configs * n_seeds, showspeed=true)
    @threads for i in 1:n_configs
        alg_params, agent_params = sample_goal_params()
        alg = PPO(; alg_params...)
        for j in 1:n_seeds
            seed = rand(UInt32)
            Random.seed!(seed)
            env = MultiThreadedParallelEnv([GoalReachingEnv() for _ in 1:16])
            policy = ActorCriticPolicy(observation_space(env), action_space(env))
            agent = ActorCriticAgent(policy; verbose=0, agent_params..., log_dir="logs/goalreaching_search/run_$(run)/trial_$(i)_seed_$(j)")

            DRiL.TensorBoardLogger.write_hparams!(agent.logger, DRiL.get_hparams(alg), ["env/avg_step_rew", "train/loss"])
            learn_stats = learn!(agent, env, alg; max_steps=100_000)
            next!(p)
        end
    end
end
##
# run_experiment(5, 5, 3)