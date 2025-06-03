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
using ClassicControlEnvironments
##
stats_window_size = 50
alg = PPO(; ent_coef=0.01f0, vf_coef=0.5f0, gamma=0.9999f0, gae_lambda=0.9f0, 
    clip_range=0.2f0,
    max_grad_norm=0.5f0,
)
env = MultiThreadedParallelEnv([MountainCarEnv() |> ScalingWrapperEnv for _ in 1:32])
env = MonitorWrapperEnv(env, stats_window_size)
env = NormalizeWrapperEnv(env, gamma=alg.gamma)

policy = ActorCriticPolicy(observation_space(env), action_space(env), log_std_init=0f0)
agent = ActorCriticAgent(policy; verbose=2, n_steps=32, learning_rate=3f-4, epochs=10,
    log_dir=logdir("mountaincar_test", "normalized_monitored_run"), batch_size=256)
DRiL.TensorBoardLogger.write_hparams!(agent.logger, alg, agent, ["env/avg_step_rew", "train/loss"])
learn_stats = learn!(agent, env, alg; max_steps=2_000_000)
## this seems ok, we get some shorter episodes
