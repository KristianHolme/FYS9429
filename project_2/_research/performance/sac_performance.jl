using DrWatson
@quickactivate :project_2
using DRiL
using WGLMakie
using ClassicControlEnvironments
using Lux: relu
using Zygote
##
ent_coef = FixedEntropyCoefficient(0.051f0)
# ent_coef = AutoEntropyCoefficient()
alg = SAC(; ent_coef, buffer_capacity=50_000, batch_size=512, train_freq=32, gradient_steps=32,
    gamma=0.9999f0, tau=0.01f0, start_steps=0)
env = BroadcastedParallelEnv([MountainCarContinuousEnv() for _ in 1:1])
env = MonitorWrapperEnv(env)
env = NormalizeWrapperEnv(env, gamma=alg.gamma)

policy = SACPolicy(observation_space(env), action_space(env);
    log_std_init=-0.22f0, hidden_dims=[64, 64])
agent = SACAgent(policy, alg; verbose=2, log_dir=logdir("sac_performance", "mountainccar_cont"))
DRiL.TensorBoardLogger.write_hparams!(agent.logger, DRiL.get_hparams(alg), ["env/ep_rew_mean", "train/actor_loss", "train/critic_loss", "train/entropy_loss"])
##
@profview agent, replay_buffer, training_stats = learn!(agent, env, alg, 1_000)
##
@code_warntype learn!(agent, replay_buffer, env, alg, 5_000)



##
obs = rand(observation_space(env), 4)
batched_obs = DRiL.batch(obs, observation_space(env))
ps = agent.train_state.parameters
st = agent.train_state.states
actor_feats, _, st = DRiL.extract_features(policy, batched_obs, ps, st)
action_means, st = DRiL.get_actions_from_features(policy, actor_feats, ps, st)
log_std = ps.log_std

batch_dim = ndims(action_means)
@code_warntype ds = SquashedDiagGaussian.(eachslice(action_means, dims=batch_dim), Ref(log_std))
##