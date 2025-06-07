using DrWatson
@quickactivate :project_2
using Lux
using DRiL
using Zygote
using WGLMakie
WGLMakie.activate!()
# using CairoMakie
using ClassicControlEnvironments
using ProgressMeter
##
stats_window_size = 50
alg = PPO(; ent_coef=0.01f0, vf_coef=0.5f0, gamma=0.9999f0, gae_lambda=0.9f0, 
    clip_range=0.2f0,
    max_grad_norm=0.5f0,
)

N_envs = 32
broadcastedenv = BroadcastedParallelEnv([MountainCarEnv() |> ScalingWrapperEnv for _ in 1:N_envs])
broadcastedenv = MonitorWrapperEnv(broadcastedenv, stats_window_size)
broadcastedenv = NormalizeWrapperEnv(broadcastedenv, gamma=alg.gamma)

multienv = MultiThreadedParallelEnv([MountainCarEnv() |> ScalingWrapperEnv for _ in 1:N_envs])
multienv = MonitorWrapperEnv(multienv, stats_window_size)
multienv = NormalizeWrapperEnv(multienv, gamma=alg.gamma)

policy = ActorCriticPolicy(observation_space(broadcastedenv), action_space(broadcastedenv))
agent = ActorCriticAgent(policy; verbose=2)

roll_buffer = RolloutBuffer(observation_space(broadcastedenv), action_space(broadcastedenv), alg.gae_lambda, alg.gamma, agent.n_steps, N_envs)

function run_env(env, agent, roll_buffer)
    fps = DRiL.collect_rollouts!(roll_buffer, agent, env)
end

@time run_env(broadcastedenv, agent, roll_buffer)
@time run_env(multienv, agent, roll_buffer)


function benchmark_parallel_envs(N_envs_vec::Vector{Int}, n_runs::Int=20)
    broadcasted_fps = Float64[]
    multienv_fps = Float64[]
    p = Progress(length(N_envs_vec)*n_runs*2)
    for N_envs in N_envs_vec
        broadcastedenv = BroadcastedParallelEnv([MountainCarEnv() |> ScalingWrapperEnv for _ in 1:N_envs])
        broadcastedenv = MonitorWrapperEnv(broadcastedenv, stats_window_size)
        broadcastedenv = NormalizeWrapperEnv(broadcastedenv, gamma=alg.gamma)

        multienv = MultiThreadedParallelEnv([MountainCarEnv() |> ScalingWrapperEnv for _ in 1:N_envs])
        multienv = MonitorWrapperEnv(multienv, stats_window_size)
        multienv = NormalizeWrapperEnv(multienv, gamma=alg.gamma)

        policy = ActorCriticPolicy(observation_space(broadcastedenv), action_space(broadcastedenv))
        agent = ActorCriticAgent(policy; verbose=2)

        roll_buffer = RolloutBuffer(observation_space(broadcastedenv), action_space(broadcastedenv), alg.gae_lambda, alg.gamma, agent.n_steps, N_envs)

        current_broadcasted_fps = Float64[]
        current_multienv_fps = Float64[]
        for _ in 1:n_runs
            fps = run_env(broadcastedenv, agent, roll_buffer)
            push!(current_broadcasted_fps, fps)
            next!(p)
        end
        push!(broadcasted_fps, mean(current_broadcasted_fps))

        for _ in 1:n_runs
            fps = run_env(multienv, agent, roll_buffer)
            push!(current_multienv_fps, fps)
            next!(p)
        end
        push!(multienv_fps, mean(current_multienv_fps))
    end
    return broadcasted_fps, multienv_fps
end
##
Ns = [1, 2, 4, 8, 16, 32, 64, 128]
broadcasted_fps, multienv_fps = benchmark_parallel_envs(Ns)
##
fig = Figure()
ax = Axis(fig[1, 1], ylabel="FPS", xlabel="Number of environments", xticks=Ns)
lines!(ax, Ns, broadcasted_fps, label="Broadcasted")
lines!(ax, Ns, multienv_fps, label="MultiThreaded")
axislegend(ax, position=:lt)
fig
