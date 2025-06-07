using DrWatson
@quickactivate :project_2
using DRiL
using ClassicControlEnvironments
using WGLMakie
using Zygote
WGLMakie.activate!()
##
env = CartPoleEnv()
##
ClassicControlEnvironments.plot(env.problem)
ClassicControlEnvironments.interactive_viz(env)

##
stats_window_size = 50
alg = PPO(; ent_coef=0f0, vf_coef=0.5f0, gamma=0.98f0, gae_lambda=0.8f0)
cartenv = MultiThreadedParallelEnv([CartPoleEnv() for _ in 1:8])
cartenv = MonitorWrapperEnv(cartenv, stats_window_size)
cartenv = NormalizeWrapperEnv(cartenv, gamma=alg.gamma)

cartpolicy = ActorCriticPolicy(observation_space(cartenv), action_space(cartenv))
cartagent = ActorCriticAgent(cartpolicy; verbose=2, n_steps=32, batch_size=256, learning_rate=3f-4, epochs=20,
    log_dir=logdir("cartpole_test", "normalized_monitored_run"))
DRiL.TensorBoardLogger.write_hparams!(cartagent.logger, alg, cartagent, ["env/ep_rew_mean", "train/loss"])

trajs = DRiL.collect_trajectories(cartagent, cartenv, 10)
trajs[1].actions
rollout_buffer = RolloutBuffer(observation_space(cartenv), action_space(cartenv), alg.gae_lambda, alg.gamma, 10, 8)
DRiL.collect_rollouts!(rollout_buffer, cartagent, cartenv)
rollout_buffer.observations
rollout_buffer.actions


obs = observe(cartenv)
actions = predict_actions(cartagent, obs)

ps = cartagent.train_state.parameters
st = cartagent.train_state.states
rng = cartagent.rng

feats, st = DRiL.extract_features(cartpolicy, obs, ps, st)
feats
action_logits, st = DRiL.get_actions_from_features(cartpolicy, feats, ps, st)  # For discrete, these are logits
action_logits
actions = DRiL.get_discrete_actions(cartpolicy, action_logits, Random.default_rng(); log_probs=false, deterministic=true)



argmax.(eachcol(action_logits))
actions, values, logprobs, st = cartpolicy(obs, ps, st)
logprobs



##
learn_stats = learn!(cartagent, cartenv, alg; max_steps=2048)

@profview learn!(cartagent, cartenv, alg; max_steps=4096)

ps = cartagent.train_state.parameters
st = cartagent.train_state.states
using BenchmarkTools
@profview @btime DRiL.evaluate_actions(policy, obs, actions, ps, st) setup = (obs = rand(Float32, 4, 8); actions = rand([1, 2], (1, 8)))

@benchmark DRiL.evaluate_actions(cartpolicy, obs, actions, ps, st) setup = (obs = rand(Float32, 4, 8); actions = rand([1, 2], (1, 8)))
"""
BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
 Range (min … max):  25.109 μs …  14.349 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     32.590 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   53.808 μs ± 509.548 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

       ▂▆▄       █▅▂   █▇▄▄▂▄▇▃▁▂▃▂                             
  ▂▂▁▂▃████▇▆▄▄▄▅████▆███████████████▇▆▆▅▅▄▄▄▆▆▅▄▅▄▄▄▃▃▃▃▃▂▂▂▂ ▅
  25.1 μs         Histogram: frequency by time         43.6 μs <

 Memory estimate: 12.34 KiB, allocs estimate: 119.
 """

## test learning with mountaincar
mountainenv = MultiThreadedParallelEnv([MountainCarEnv() for _ in 1:8])
mountainenv = MonitorWrapperEnv(mountainenv, stats_window_size)
mountainenv = NormalizeWrapperEnv(mountainenv, gamma=alg.gamma)
mountainpolicy = ActorCriticPolicy(observation_space(mountainenv), action_space(mountainenv))
mountainagent = ActorCriticAgent(mountainpolicy; verbose=2, n_steps=32, batch_size=256, learning_rate=3f-4, epochs=20,
    log_dir=logdir("mountaincar_test", "normalized_monitored_run"))
DRiL.TensorBoardLogger.write_hparams!(mountainagent.logger, alg, mountainagent, ["env/ep_rew_mean", "train/loss"])


@time learn!(mountainagent, mountainenv, alg; max_steps=100_000)
##
ps = mountainagent.train_state.parameters
st = mountainagent.train_state.states
@benchmark DRiL.evaluate_actions(policy, obs, actions, ps, st) setup = (obs = rand(Float32, 2, 8); actions = rand(1, 8))
"""
BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
 Range (min … max):  14.834 μs …   9.052 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     26.771 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   30.216 μs ± 118.466 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

                     ▄█▇▃                                       
  ▁▁▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▄████▇▅▅▄▅▇██▇▆▅▃▃▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  14.8 μs         Histogram: frequency by time         44.8 μs <

 Memory estimate: 11.28 KiB, allocs estimate: 88.
 """
##
@profview learn!(agent, mountainenv, alg; max_steps=4096)


action_logits = predict_actions(mountainagent, obs, deterministic=true)

##
using Lux
using Distributions
using BenchmarkTools
logits = rand(Float32, 2, 8)
probs = Lux.softmax(logits)
distributions = Distributions.Categorical.(eachcol(probs) .|> Vector)
actions = rand(1:2, 8)

function distr_log_probs_and_entropy(logits, actions)
    probs = Lux.softmax(logits)
    distributions = Distributions.Categorical.(eachcol(probs) .|> Vector)
    log_probs = loglikelihood.(distributions, actions)
    entropy = Distributions.entropy.(distributions)
    return log_probs, entropy
end

distr_log_probs_and_entropy(logits, actions)
@benchmark distr_log_probs_and_entropy(logits, actions) setup = (logits = rand(Float32, 2, 8); actions = rand(1:2, 8))
"""
BenchmarkTools.Trial: 10000 samples with 9 evaluations per sample.
 Range (min … max):  2.297 μs …  13.580 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     3.984 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   4.006 μs ± 756.874 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

               ▆▆▄▂       ▃█▆▅▄▂                               
  ▁▁▁▁▁▂▂▂▃▂▃▃▆██████▇▆▇▇▇███████▇▅▅▅▄▄▄▃▃▃▃▃▃▃▂▃▂▂▂▂▁▂▂▂▁▁▁▁ ▃
  2.3 μs          Histogram: frequency by time        6.24 μs <

 Memory estimate: 1.62 KiB, allocs estimate: 42.
 """
log_probs, entropy = distr_log_probs_and_entropy(logits, actions)


logits = rand(Float32, 2, 8)
actions = rand(1:2, (1, 8))
DRiL.get_discrete_logprobs_and_entropy(logits, actions)[1]







@benchmark DRiL.get_discrete_logprobs_and_entropy(logits, actions) setup = (logits = rand(Float32, 2, 8); actions = rand(1:2, 8))




##
using Distributions
log_probs = loglikelihood.(distributions, actions)
log_probs

m = rand(2, 8)
log_prob_mat = Lux.logsoftmax(m)
actions = rand(1:2, 8)
actions = reshape(actions, 1, 8)
log_probs = getindex.(eachcol(log_prob_mat), actions)
log_probs






##