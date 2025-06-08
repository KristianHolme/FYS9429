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
stats_window_size = 50
alg = PPO(; ent_coef=0.008623f0, vf_coef=0.2607f0, gamma=0.994528f0, gae_lambda=0.871101f0, 
    clip_range=0.273062f0,
    max_grad_norm=0.5f0,
)
env = BroadcastedParallelEnv([MountainCarEnv() for _ in 1:16])
env = MonitorWrapperEnv(env, stats_window_size)
env = NormalizeWrapperEnv(env, gamma=alg.gamma)

policy = ActorCriticPolicy(observation_space(env), action_space(env), log_std_init=-0.145555f0)
agent = ActorCriticAgent(policy; verbose=2, n_steps=128, batch_size=64, epochs=10,
    log_dir=logdir("mountaincar_test", "normalized_monitored_run"))
DRiL.TensorBoardLogger.write_hparams!(agent.logger, alg, agent, ["env/ep_rew_mean", "train/loss"])
##
learn!(agent, env, alg; max_steps=10_000)
@profview learn_stats = learn!(agent, env, alg; max_steps=80_000)
## this seems ok, we get some shorter episodes


single_env = MountainCarEnv()
obs, actions, rewards = collect_trajectory(agent, single_env; norm_env=env)
sum(rewards)
fig_traj = plot_trajectory(MountainCarEnv(), obs, actions, rewards)
plot_trajectory_interactive(MountainCarEnv(), obs, actions, rewards)