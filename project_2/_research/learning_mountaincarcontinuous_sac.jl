using DrWatson
@quickactivate :project_2
using DRiL
using WGLMakie
using ClassicControlEnvironments
using Lux: relu
using Zygote
##
# ent_coef = FixedEntropyCoefficient(0.1f0)
ent_coef = AutoEntropyCoefficient()
alg = SAC(; ent_coef, buffer_capacity=50_000, batch_size=512, train_freq=32, gradient_steps=32,
    gamma=0.9999f0, tau=0.01f0, start_steps=0)
env = BroadcastedParallelEnv([MountainCarContinuousEnv() for _ in 1:1])
env = MonitorWrapperEnv(env)
env = NormalizeWrapperEnv(env, gamma=alg.gamma)

policy = ContinuousActorCriticPolicy(observation_space(env), action_space(env),
    activation=relu, critic_type=QCritic(), log_std_init=1f0)
agent = SACAgent(policy, alg; verbose=2, log_dir=logdir("mountaincar_sac_test", "normalized_monitored_run"))
DRiL.TensorBoardLogger.write_hparams!(agent.logger, DRiL.get_hparams(alg), ["env/ep_rew_mean", "train/loss"])
##
agent, replay_buffer, training_stats = learn!(agent, env, alg, 50_000)
##
extrema(replay_buffer.rewards)
actions = getindex.(replay_buffer.actions, 1)
extrema(actions)
##
single_env = MountainCarContinuousEnv()
obs, actions, rewards = collect_trajectory(agent, single_env; norm_env=env, deterministic=true)
sum(rewards)
fig_traj = plot_trajectory(single_env, obs, actions, rewards)
## plot actions against position and velocity
vec_actions = getindex.(actions, 1)
vec_pos = getindex.(obs, 1)
vec_vel = getindex.(obs, 2)
fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, vec_pos[1:length(vec_actions)], vec_actions)
ax2 = Axis(fig[2, 1])
lines!(ax2, vec_vel[1:length(vec_actions)], vec_actions)
fig
## action histogram
hist(vec_actions)
##
plot_trajectory_interactive(single_env, obs, actions, rewards)
##
ps = agent.train_state.parameters
ps.log_std
keys(ps)