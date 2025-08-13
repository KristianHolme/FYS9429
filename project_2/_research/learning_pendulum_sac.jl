using DrWatson
@quickactivate :project_2
using DRiL
using WGLMakie
using ClassicControlEnvironments
using Lux: relu
using Zygote
##
alg = SAC(; learning_rate=1f-3)
env = BroadcastedParallelEnv([PendulumEnv() for _ in 1:1])
env = MonitorWrapperEnv(env)
env = NormalizeWrapperEnv(env, gamma=alg.gamma)

policy = SACPolicy(observation_space(env), action_space(env))
agent = SACAgent(policy, alg; verbose=2, log_dir=logdir("pendulum_sac_test", "test"))
DRiL.TensorBoardLogger.write_hparams!(agent.logger, DRiL.get_hparams(alg), ["env/ep_rew_mean", "train/actor_loss", "train/critic_loss", "train/entropy_loss"])
##
agent, replay_buffer, training_stats = learn!(agent, env, alg, 20_000)
##
extrema(replay_buffer.rewards)
actions = getindex.(replay_buffer.actions, 1)
extrema(actions)
##
single_env = PendulumEnv()
obs, actions, rewards = collect_trajectory(agent, single_env; norm_env=env, deterministic=true)
sum(rewards)
fig_traj = plot_trajectory(single_env, obs, actions, rewards)
display(fig_traj)
## plot actions against position and velocity
vec_actions = getindex.(actions, 1)
vec_pos = getindex.(obs, 1)
vec_vel = getindex.(obs, 2)
fig = Figure()
ax = Axis(fig[1, 1], xlabel="position", ylabel="action")
lines!(ax, vec_pos[1:length(vec_actions)], vec_actions)
ax2 = Axis(fig[2, 1], xlabel="velocity", ylabel="action")
lines!(ax2, vec_vel[1:length(vec_actions)], vec_actions)
fig
## action histogram
hist(vec_actions)
##
fig, _ = plot_trajectory_interactive(single_env, obs, actions, rewards)
display(fig)
##
ps = agent.train_state.parameters
ps.log_std
keys(ps)