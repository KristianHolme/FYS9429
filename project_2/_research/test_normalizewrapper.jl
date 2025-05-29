using DrWatson
@quickactivate :project_2
using Lux
using DRiL
using Zygote
using WGLMakie
WGLMakie.activate!()
using Statistics
using LinearAlgebra
using ClassicControlEnvironments
##
struct NotNormalizedEnv <: AbstractEnv end
DRiL.observation_space(env::NotNormalizedEnv) = UniformBox{Float32}(50.0f0, 60.0f0, (3,))
DRiL.action_space(env::NotNormalizedEnv) = UniformBox{Float32}(7f0, 9f0, (1,))
DRiL.observe(env::NotNormalizedEnv) = rand(observation_space(env))
DRiL.terminated(env::NotNormalizedEnv) = false
DRiL.truncated(env::NotNormalizedEnv) = false
DRiL.act!(env::NotNormalizedEnv, action) = randn() * 20f0 + 1000f0
DRiL.get_info(env::NotNormalizedEnv) = Dict()
DRiL.reset!(env::NotNormalizedEnv) = nothing
##
env = MultiThreadedParallelEnv([NotNormalizedEnv() for _ in 1:4])
env = NormalizeWrapperEnv(env)
##
all_obs = Vector{Float32}[]
all_rewards = Float32[]
for _ in 1:10000
    rewards, term, trunc, info = DRiL.step!(env, rand(action_space(env), 4))
    push!(all_rewards, rewards...)
    push!(all_obs, eachcol(observe(env))...)
end
##
getindex.(all_obs, 1) |> extrema


##
fig = Figure()
ax_ob1 = Axis(fig[1, 1], title="obs 1")
ax_ob2 = Axis(fig[1, 2], title="obs 2")
ax_ob3 = Axis(fig[2, 1], title="obs 3")
ax_rew = Axis(fig[2, 2], title="rew")
hist!(ax_ob1, getindex.(all_obs, 1), bins=100)
hist!(ax_ob2, getindex.(all_obs, 2), bins=100)
hist!(ax_ob3, getindex.(all_obs, 3), bins=100)
hist!(ax_rew, all_rewards, bins=100)
fig
##
getindex.(all_obs, 1) |> mean
getindex.(all_obs, 1) |> std
getindex.(all_obs, 2) |> mean
getindex.(all_obs, 2) |> std
getindex.(all_obs, 3) |> mean
getindex.(all_obs, 3) |> std
##






##
normenv.obs_mean
normenv.obs_std
normenv.obs_mean
##