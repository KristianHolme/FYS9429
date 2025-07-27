using DrWatson
@quickactivate :project_2
using DRiL
using ClassicControlEnvironments
using Zygote
using BenchmarkTools
##
env = CartPoleEnv()

act_space = action_space(env)
stats_window_size = 50
alg = PPO(; ent_coef=0f0, vf_coef=0.5f0, gamma=0.98f0, gae_lambda=0.8f0)
multenv = MultiThreadedParallelEnv([CartPoleEnv() for _ in 1:8])
monenv = MonitorWrapperEnv(multenv, stats_window_size)
normenv = NormalizeWrapperEnv(monenv, gamma=alg.gamma)

policy = ActorCriticPolicy(observation_space(normenv), action_space(normenv))
agent = ActorCriticAgent(policy; verbose=2, n_steps=32, batch_size=256, learning_rate=3f-4, epochs=20)
roll_buffer = RolloutBuffer(observation_space(normenv), action_space(normenv), alg.gae_lambda, alg.gamma, 10, 8)



function run_env(env, agent, roll_buffer)
    fps = DRiL.collect_rollout!(roll_buffer, agent, env)
end

run_env(multenv, agent, roll_buffer)
run_env(monenv, agent, roll_buffer)
run_env(normenv, agent, roll_buffer)

@benchmark run_env($multenv, $agent, $roll_buffer)
@benchmark run_env($monenv, $agent, $roll_buffer)
@benchmark run_env($normenv, $agent, $roll_buffer)

@profview begin
    for i in 1:5_000
        run_env(multenv, agent, roll_buffer)
    end
end
@profview begin
    for i in 1:5_000
        run_env(monenv, agent, roll_buffer)
    end
end
@profview begin
    for i in 1:5_000
        run_env(normenv, agent, roll_buffer)
    end
end


##