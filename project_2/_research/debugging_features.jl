using DrWatson
@quickactivate :project_2
using DRiL
using Random
using Lux
##
observation_space = DRiL.Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
action_space = DRiL.Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])

rng = Random.Xoshiro(42)
policy = ContinuousActorCriticPolicy(observation_space, action_space; hidden_dims=[4, 4], critic_type=QCritic())
ps = Lux.initialparameters(rng, policy)
st = Lux.initialstates(rng, policy)

batch_size = 4
obs = rand(rng, observation_space, batch_size)
batch_obs = DRiL.batch(obs, observation_space)
actor_feats, critic_feats, st = DRiL.extract_features(policy, batch_obs, ps, st)
actor_feats
critic_feats
##
actions, new_st = DRiL.get_actions_from_features(policy, actor_feats, ps, st)
actions, actor_st = policy.actor_head[1](actor_feats, ps.actor_head[:layer_1], st.actor_head[:layer_1])
actions