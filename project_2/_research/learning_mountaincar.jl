using DrWatson
@quickactivate :project_2
using DRiL
using WGLMakie
using ClassicControlEnvironments
##
alg = PPO(;ent_coef=0.01f0)
env = BroadcastedParallelEnv([MountainCarEnv() for _ in 1:16])
env = MonitorWrapperEnv(env)
env = NormalizeWrapperEnv(env, gamma=alg.gamma)

policy = ActorCriticPolicy(observation_space(env), action_space(env))
agent = ActorCriticAgent(policy; verbose=2, n_steps=256, batch_size=64, epochs=10,
    log_dir=logdir("mountaincar_test", "normalized_monitored_run"))
DRiL.TensorBoardLogger.write_hparams!(agent.logger, alg, agent, ["env/ep_rew_mean", "train/loss"])
##
learn!(agent, env, alg; max_steps=100_000)
##
single_env = MountainCarEnv()
obs, actions, rewards = collect_trajectory(agent, single_env; norm_env=env)
sum(rewards)
##
fig_traj = plot_trajectory(single_env, obs, actions, rewards)
plot_trajectory_interactive(single_env, obs, actions, rewards)
