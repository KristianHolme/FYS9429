using DrWatson
@quickactivate :project_2
using DataFrames
using ClassicControlEnvironments
# using WGLMakie
# WGLMakie.activate!()
using AlgebraOfGraphics
using Statistics
##
using CairoMakie
CairoMakie.activate!()
##
experiment_name = "ppo_search_2025-05-30_16-47"
df, best_config = analyze_results(experiment_name)
##
T = Float32
env = MultiThreadedParallelEnv([PendulumEnv() |> ScalingWrapperEnv for _ in 1:best_config[:n_envs]])
policy = ActorCriticPolicy(observation_space(env), action_space(env))
agent = ActorCriticAgent(policy; verbose=2, n_steps=best_config[:n_steps], learning_rate=T(best_config[:learning_rate]), epochs=best_config[:epochs],
    log_dir="logs/working_tests/best_config", batch_size=best_config[:batch_size])
alg = PPO(; ent_coef=T(best_config[:ent_coef]), vf_coef=T(best_config[:vf_coef]), gamma=T(best_config[:gamma]), gae_lambda=T(best_config[:gae_lambda]))
DRiL.TensorBoardLogger.write_hparams!(agent.logger, alg, agent, ["env/avg_step_rew", "train/loss"])
learn_stats = learn!(agent, env, alg; max_steps=100_000)
##
using JLD2
jldsave("data/best_config.jld2"; agent, learn_stats)
agent = load("data/best_config.jld2")["agent"]
##
single_env = PendulumEnv() |> ScalingWrapperEnv
observations, actions, rewards = collect_trajectory(agent, single_env)
actions = first.(actions) .* 2
mean(rewards)
sum(rewards)
plot_trajectory_interactive(PendulumEnv(), observations, actions, rewards)
animate_trajectory_video(PendulumEnv(), observations, actions, "test.mp4")
##
N = 1000
undiscounted_returns = zeros(N)
for i in 1:N
    observations, actions, rewards = collect_trajectory(agent, single_env)
    undiscounted_returns[i] = sum(rewards)
end
mean(undiscounted_returns)
std(undiscounted_returns)
hist(undiscounted_returns)
##

# Convert columns to concrete types to avoid AbstractFloat issues
for col in names(df)
    if eltype(df[!, col]) <: AbstractFloat
        df[!, col] = Float64.(df[!, col])
    end
end

columns = names(df)
fig_opts = (size=(1000, 600),)

datacols = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
## giant cross plot
fig = Figure(size=(2000, 2000))
for (ix, i) in enumerate(datacols), (jx, j) in enumerate(datacols)
    plt = data(df) * mapping(Symbol(columns[i]), Symbol(columns[j]), color=:mean_return) *
          (visual(Scatter) + smooth())
    draw!(fig[jx, ix], plt, facet=(; linkxaxes=:minimal, linkyaxes=:minimal))
end
fig
##
fig = Figure(size=(1000, 1000))
for (ix, i) in enumerate(datacols)
    row = div(ix - 1, 2) + 1
    col = mod1(ix, 2)
    plt = data(df) * mapping(Symbol(columns[i]), :mean_return) *
          (visual(Scatter) + smooth())
    draw!(fig[row, col], plt)
end
fig



##
x_data = Symbol(columns[datacols[7]])
data(df) * mapping(x_data, :mean_return, color=:gamma) *
(visual(Scatter) + smooth()) |> draw(figure=fig_opts, axis=())

##
x_data = :batch_size
data(df) * mapping(x_data, :mean_return,
    row=:batch_size => nonnumeric,
    col=:epochs => nonnumeric,
    color=:gamma,
    marker=:n_steps => nonnumeric,
) * visual(Scatter) |> draw(figure=fig_opts)