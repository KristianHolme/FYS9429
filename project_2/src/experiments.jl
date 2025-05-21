function default_PPO(env::PendulumEnv)
    PPO(; gamma=0.9f0,
        gae_lambda=0.95f0)
end
function default_agent(env::PendulumEnv; folder="hyper_search", name="run")
    policy = ActorCriticPolicy(observation_space(env), action_space(env))
    agent = ActorCriticAgent(policy; verbose=2, n_steps=256, learning_rate=1f-3, epochs=10,
        log_dir=joinpath("logs", folder, name), batch_size=64)
    return agent
end

function default_env(; n_envs::Int=16)
    env = MultiThreadedParallelEnv([PendulumEnv() |> ScalingWrapperEnv for _ in 1:n_envs])
    return env
end