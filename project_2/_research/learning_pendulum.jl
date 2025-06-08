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
alg = PPO(; ent_coef=0f0, vf_coef=0.480177f0, gamma=0.990886f0, gae_lambda=0.85821f0, clip_range=0.132141f0)
pendenv = BroadcastedParallelEnv([PendulumEnv() |> ScalingWrapperEnv for _ in 1:8])
pendenv = MonitorWrapperEnv(pendenv, stats_window_size)
pendenv = NormalizeWrapperEnv(pendenv, gamma=alg.gamma)

pendpolicy = ActorCriticPolicy(observation_space(pendenv), action_space(pendenv))
pendagent = ActorCriticAgent(pendpolicy; verbose=2, n_steps=128, batch_size=128, learning_rate=1.95409f-4, epochs=20,
    log_dir=logdir("pendulum_test", "normalized_monitored_run"))
DRiL.TensorBoardLogger.write_hparams!(pendagent.logger, alg, pendagent, ["env/ep_rew_mean", "train/loss"])
##
learn_stats = learn!(pendagent, pendenv, alg; max_steps=5_000)
##
single_env = PendulumEnv()
obs, actions, rewards = collect_trajectory(pendagent, single_env; norm_env=pendenv)
actions
sum(rewards)
fig_traj = plot_trajectory(PendulumEnv(), obs, actions, rewards)
plot_trajectory_interactive(PendulumEnv(), obs, actions, rewards)

