using DrWatson
@quickactivate :project_2
using Lux
using DRiL
using Zygote
using WGLMakie
WGLMakie.activate!()
# using CairoMakie
using Statistics
using LinearAlgebra
using Pendulum
##
env = MultiThreadedParallelEnv([PendulumEnv() |> ScalingWrapperEnv for _ in 1:16])
policy = ActorCriticPolicy(observation_space(env), action_space(env))
agent = ActorCriticAgent(policy; verbose=2, n_steps=256, learning_rate=1f-3, epochs=20,
    log_dir="logs/working_tests/run", batch_size=128)
alg = PPO(; ent_coef=0.001f0, vf_coef=0.5f0, gamma=0.98f0, gae_lambda=0.95f0)
DRiL.TensorBoardLogger.write_hparams!(agent.logger, alg, agent, ["env/avg_step_rew", "train/loss"])
learn_stats = learn!(agent, env, alg; max_steps=100_000)
##
using JLD2
jldsave("data/saved_agent.jld2"; agent)
agent = load("data/saved_agent.jld2")["agent"]
## viz
single_env = PendulumEnv(;gravity=10f0) |> ScalingWrapperEnv
observations, actions, rewards = collect_trajectory(agent, single_env)
actions = first.(actions) .* 2  
mean(rewards)
plot_trajectory_interactive(PendulumEnv(), observations, actions, rewards)