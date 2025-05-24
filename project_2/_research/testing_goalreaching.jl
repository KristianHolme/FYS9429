using DrWatson
@quickactivate :project_2
using Lux
using DRiL
using Zygote
using CairoMakie
using Accessors
##
env = MultiThreadedParallelEnv([GoalReachingEnv() for _ in 1:16])
policy = ActorCriticPolicy(observation_space(env), action_space(env))
agent = ActorCriticAgent(policy; verbose=2, n_steps=256, learning_rate=3f-4, epochs=10,
    log_dir="logs/goalreaching/run", batch_size=64)
alg = PPO(; ent_coef=0.01f0, vf_coef=0.5f0, gamma=0.99f0, gae_lambda=0.9f0)
DRiL.TensorBoardLogger.write_hparams!(agent.logger, alg, agent, ["env/avg_step_rew", "train/loss"])
@reset agent.learning_rate = 3f-5
learn_stats = learn!(agent, env, alg; max_steps=300_000)
##
single_env = GoalReachingEnv()
observations, actions, rewards = collect_trajectory(agent, single_env)
player_positions = getindex.(observations, 1)
goal_positions = getindex.(observations, 2)

fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1])
lines!(ax, player_positions, label="Player")
lines!(ax, goal_positions, label="Goal")
axislegend(ax, position=:rt)
fig

##
n=500
ppos = rand(Float32, n).*2 .- 1
gpos = rand(Float32, n).*2 .- 1
observations = [ppos gpos]'

values = predict_values(agent, observations)

fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1])
lines!(ax, ppos, label="Player")
lines!(ax, gpos, label="Goal")
axislegend(ax, position=:rt)
fig