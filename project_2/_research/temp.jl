using DrWatson
@quickactivate :project_2
using Lux
using DRiL
using Zygote
using WGLMakie
WGLMakie.activate!()
# using CairoMakie
using Statistics
using LinearAlgebra
using BenchmarkTools
using ClassicControlEnvironments
using ComponentArrays
using DataStructures
##
# Create environment that gives reward 1.0 only at final step
env = BroadcastedParallelEnv([MountainCarContinuousEnv()])
policy = ContinuousActorCriticPolicy(observation_space(env), action_space(env), activation=relu, shared_features=false, critic_type=QCritic())
entropy_coefficient = AutoEntropyCoefficient()
alg = SAC(ent_coef=entropy_coefficient)
agent = SACAgent(policy, alg)

ps = agent.train_state.parameters

ps2 = copy(ps)

##
v = [rand([true, false]) for _ in 1:10]
count(v)
count(!, v)





cb = CircularBuffer{Union{Nothing,Bool}}(10)
push!(cb, true)
v = Vector{Matrix{Float32}}(undef, 10)
any(x -> isassigned(v, x), eachindex(v))
isassigned(Ref(v), eachindex(v))
any(isassigned, v)
v[end] = true