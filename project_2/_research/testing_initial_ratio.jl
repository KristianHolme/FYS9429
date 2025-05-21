trajs = DRiL.collect_trajectories(agent, env, 7)
traj = trajs[1]
obs = stack(traj.observations)
act = stack(traj.actions)
logprobs = traj.logprobs
vals = traj.values
rew = traj.rewards
new_vals, new_logprobs, new_entropy, st = DRiL.evaluate_actions(policy, obs, act, agent.train_state.parameters, agent.train_state.states)
vals
new_vals
norm(vec(vals) - vec(new_vals))
logprobs
new_logprobs
logprobs - vec(new_logprobs)
##
rng = Random.default_rng()
Random.seed!(rng, 1)
policy = agent.policy
ps = agent.train_state.parameters
st = agent.train_state.states
observations = rand(rng, Float32, 3, 7) .* 2f0 .- 1f0
# actions, values, logprobs = policy(observations, ps, st)
#Collecting trajectories:
feats, st = DRiL.extract_features(policy, observations, ps, st)
action_means, st = DRiL.get_action_mean_from_latent(policy, feats, ps, st)
values, critic_st = policy.critic_head(feats, ps.critic_head, st.critic_head)
std = exp.(ps.log_std)
Random.seed!(rng, 1234)
noise = randn(rng, Float32, 1, 7)
actions = action_means + std .* noise
distributions = DRiL.get_distributions(policy, action_means, std)
log_probs = DRiL.loglikelihood.(distributions, actions)
Random.seed!(rng, 1234)
noisy_actions, noisy_log_probs = DRiL.get_noisy_actions(policy, action_means, std, rng; log_probs=true)

new_vals, new_logprobs, new_entropy, st = DRiL.evaluate_actions(policy, observations, actions, ps, st)
logprobs
new_logprobs
vec(logprobs - new_logprobs)
##
n_steps = 7
n_envs = env.n_envs
roll_buffer = RolloutBuffer(observation_space(env), action_space(env), alg.gae_lambda, alg.gamma, n_steps, n_envs)
DRiL.collect_rollouts!(roll_buffer, agent, env)
obs = roll_buffer.observations
act = roll_buffer.actions
logprobs = roll_buffer.logprobs
vals = roll_buffer.values
rew = roll_buffer.rewards

new_vals, new_logprobs, new_entropy, st = DRiL.evaluate_actions(policy, obs, act, agent.train_state.parameters, agent.train_state.states)
vec(vals)
vec(new_vals)
norm(vec(vals) - vec(new_vals))
vec(logprobs)
vec(new_logprobs)
logprobs - vec(new_logprobs)