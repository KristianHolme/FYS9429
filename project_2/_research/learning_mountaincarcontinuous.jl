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
using ClassicControlEnvironments
##
alg = PPO(;ent_coef=0.1f0)
env = BroadcastedParallelEnv([MountainCarContinuousEnv() for _ in 1:8])
env = MonitorWrapperEnv(env)
env = NormalizeWrapperEnv(env, gamma=alg.gamma)

policy = ActorCriticPolicy(observation_space(env), action_space(env))
agent = ActorCriticAgent(policy; verbose=2, n_steps=256, batch_size=64, epochs=10,
    log_dir=logdir("mountaincar_test", "normalized_monitored_run"))
DRiL.TensorBoardLogger.write_hparams!(agent.logger, alg, agent, ["env/ep_rew_mean", "train/loss"])
##
learn!(agent, env, alg; max_steps=100_000)
##
single_env = MountainCarContinuousEnv()
obs, actions, rewards = collect_trajectory(agent, single_env; norm_env=env, deterministic=false)
sum(rewards)
fig_traj = plot_trajectory(single_env, obs, actions, rewards)
plot_trajectory_interactive(single_env, obs, actions, rewards)