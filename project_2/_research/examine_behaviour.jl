using DrWatson
@quickactivate :project_2
using DRiL
using DataFrames
using Pendulum
##
study_name = "hyper_search_second"
folder = datadir(study_name)
df = collect_results(folder)
##
df.total_reward = map(eachrow(df)) do row
    agent = row.agent
    env = PendulumEnv() |> ScalingWrapperEnv
    observations, actions, rewards = collect_trajectory(agent, env)
    sum(rewards)
end
##
best_row = sort(df, :total_reward, rev=true)[1, :]
@info "Best configuration:" best_row
##
agent = best_row.agent
env = PendulumEnv() |> ScalingWrapperEnv
observations, actions, rewards = collect_trajectory(agent, env)
actions = first.(actions) .* 2
plot_trajectory(PendulumEnv(), observations, actions, rewards)
sum(rewards)
animate_trajectory_video(PendulumEnv(), observations, actions, "test.mp4")






