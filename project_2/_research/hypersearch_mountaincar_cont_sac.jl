using DrWatson
@quickactivate :project_2
using DRiL
using WGLMakie
using ClassicControlEnvironments
using Lux: relu
using Zygote
using Base.Threads
##

fixed_ent_coefs = logrange(0.001f0, 0.1f0, 15) .|> Float32
agents = Vector{SACAgent}(undef, length(fixed_ent_coefs))
@threads for i in eachindex(fixed_ent_coefs)
    ent_coef = FixedEntropyCoefficient(fixed_ent_coefs[i])
    # ent_coef = AutoEntropyCoefficient()
    alg = SAC(; ent_coef, buffer_capacity=50_000, batch_size=512, train_freq=32, gradient_steps=32,
        gamma=0.9999f0, tau=0.01f0, start_steps=0)
    env = BroadcastedParallelEnv([MountainCarContinuousEnv() for _ in 1:1])
    env = MonitorWrapperEnv(env)
    env = NormalizeWrapperEnv(env, gamma=alg.gamma)

    policy = SACPolicy(observation_space(env), action_space(env);
        log_std_init=-0.22f0, hidden_dims=[64, 64])
    agent = SACAgent(policy, alg; verbose=2, log_dir=logdir("mountaincar_sac_test", "normalized_monitored_run"))
    DRiL.TensorBoardLogger.write_hparams!(agent.logger, DRiL.get_hparams(alg), ["env/ep_rew_mean", "train/loss"])
    ##
    agent, replay_buffer, training_stats = learn!(agent, env, alg, 15_000)
    agents[i] = agent
end
agent = agents[13]
fixed_ent_coefs[13]