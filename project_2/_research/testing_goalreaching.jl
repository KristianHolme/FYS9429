using DrWatson
@quickactivate :project_2
using Lux
using DRiL
using Zygote
using CairoMakie
using Accessors
using IntervalSets
using Random
##
env = BroadcastedParallelEnv([GoalReachingEnv() for _ in 1:16])
env = MonitorWrapperEnv(env)
policy = ActorCriticPolicy(observation_space(env), action_space(env))
alg = PPO(; ent_coef=0.005f0, vf_coef=0.5f0, gamma=0.99f0, gae_lambda=0.85f0,
    n_steps=256, learning_rate=3f-5, epochs=10, batch_size=64)
agent = ActorCriticAgent(policy, alg; verbose=2,
    log_dir="logs/goalreaching/run")
DRiL.TensorBoardLogger.write_hparams!(agent.logger, DRiL.get_hparams(alg), ["env/avg_step_rew", "train/loss"])
learn_stats = learn!(agent, env, alg; max_steps=500_000)
##
single_env = GoalReachingEnv()
observations, actions, rewards = collect_trajectory(agent, single_env)
player_positions = getindex.(observations, 1)
goal_positions = getindex.(observations, 2)

fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1], limits=(nothing, (-1, 1)))
error_ax = Axis(fig[1, 2], yscale=log10)
lines!(ax, player_positions, label="Player")
lines!(ax, goal_positions, label="Goal")
lines!(error_ax, abs.(player_positions .- goal_positions), label="Error")
axislegend(ax, position=:rt)
fig

##
function optimal_action(obs)
    ppos, gpos = obs
    ppos = ppos[1]
    gpos = gpos[1]
    if ppos > gpos
        return -(ppos - gpos) / (ppos + 1)
    else
        return (gpos - ppos) / (1 - ppos)
    end
end
##
o = observe(single_env)
optimal_action(o)
a = predict_actions(agent, o, deterministic=true)
r = act!(single_env, a)

##
n = 500
ppos = rand(Float32, n) .* 2 .- 1
gpos = rand(Float32, n) .* 2 .- 1
observations = [ppos gpos]'

values = predict_values(agent, observations)

fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1])
lines!(ax, ppos, label="Player")
lines!(ax, gpos, label="Goal")
axislegend(ax, position=:rt)
fig