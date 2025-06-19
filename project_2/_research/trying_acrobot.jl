using DrWatson
@quickactivate :project_2
using DRiL
using ClassicControlEnvironments
using WGLMakie
using Zygote
WGLMakie.activate!()
##
env = AcrobotEnv()
##
# using CairoMakie
# CairoMakie.activate!()
ClassicControlEnvironments.plot(env.problem)
ClassicControlEnvironments.interactive_viz(env)

##
stats_window_size = 50
alg = PPO()
acrobotenv = BroadcastedParallelEnv([AcrobotEnv() for _ in 1:8])
acrobotenv = MonitorWrapperEnv(acrobotenv, stats_window_size)
acrobotenv = NormalizeWrapperEnv(acrobotenv, gamma=alg.gamma)

acrobotpolicy = ActorCriticPolicy(observation_space(acrobotenv), action_space(acrobotenv))
acrobotagent = ActorCriticAgent(acrobotpolicy; verbose=2, n_steps=128, batch_size=128, learning_rate=3f-4, epochs=10,
    log_dir=logdir("acrobot_test", "normalized_monitored_run"))
DRiL.TensorBoardLogger.write_hparams!(acrobotagent.logger, alg, acrobotagent, ["env/ep_rew_mean", "train/loss"])
##
learn_stats = learn!(acrobotagent, acrobotenv, alg; max_steps=100_000)
##
single_env = AcrobotEnv()
obs, actions, rewards = collect_trajectory(acrobotagent, single_env; norm_env=acrobotenv)
sum(rewards)
fig_traj = plot_trajectory(AcrobotEnv(), obs, actions, rewards)
plot_trajectory_interactive(AcrobotEnv(), obs, actions, rewards)
