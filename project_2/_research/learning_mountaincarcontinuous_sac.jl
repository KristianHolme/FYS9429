using DrWatson
@quickactivate :project_2
using DRiL
using WGLMakie
using ClassicControlEnvironments
using Lux: relu
using Zygote
##
ent_coef = AutoEntropyCoefficient()
alg = SAC(; ent_coef)
env = BroadcastedParallelEnv([MountainCarContinuousEnv() for _ in 1:8])
env = MonitorWrapperEnv(env)
env = NormalizeWrapperEnv(env, gamma=alg.gamma)

policy = ContinuousActorCriticPolicy(observation_space(env), action_space(env), activation=relu, critic_type=QCritic())
agent = SACAgent(policy, alg; verbose=2, log_dir=logdir("mountaincar_sac_test", "normalized_monitored_run"))
DRiL.TensorBoardLogger.write_hparams!(agent.logger, DRiL.get_hparams(alg), ["env/ep_rew_mean", "train/loss"])
##
replay_buffer, training_stats = learn!(agent, env, alg, 100_000)
##
single_env = MountainCarContinuousEnv()
obs, actions, rewards = collect_trajectory(agent, single_env; norm_env=env, deterministic=false)
sum(rewards)
fig_traj = plot_trajectory(single_env, obs, actions, rewards)
plot_trajectory_interactive(single_env, obs, actions, rewards)

##
ps = DRiL.init_entropy_coefficient(ent_coef)
ps.log_ent_coef
keys(ps)