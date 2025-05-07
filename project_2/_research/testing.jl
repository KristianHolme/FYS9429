using DrWatson
@quickactivate :project_2
using WGLMakie
##
problem = PendulumProblem(dt=0.01f0)
pend_env = PendulumEnv()
θ, τ, fig, update_viz! = live_pendulum_viz(pend_env.problem)
act!(pend_env, 0.0f0)
update_viz!(pend_env.problem)
for i in 1:200
    act!(pend_env, rand(Float32)*4f0 - 2f0)
    update_viz!(pend_env.problem)
    sleep(pend_env.problem.dt)
end

##
problem = PendulumProblem(dt=0.01f0)
env = PendulumEnv()
θ, τ, dt, fig, sg = interactive_viz(env)


##
pend_env = PendulumEnv()
penv = MultiThreadedParallelEnv([PendulumEnv(), PendulumEnv()])
obs_space = observation_space(penv)
act_space = action_space(penv)
DRiL.reset!(penv, MersenneTwister(1234))
get_info(penv)
rewards, terminateds, truncateds, infos = DRiL.step!(penv, rand(act_space.type, act_space.shape..., 2))

## DataLoader
using MLUtils
n_steps = 12
n_envs = 2
rb = RolloutBuffer(obs_space, act_space, n_steps, n_envs, 0.97, 0.99)
DRiL.reset!(rb)

dl = DataLoader((rb.observations, rb.actions, rb.rewards, rb.advantages, rb.returns, rb.logprobs, rb.values), batchsize=3)

## Lux networks
using Lux

feat_extr = Dense(4, 64, relu)
actor = Chain(Dense(64, 64, relu), Dense(64, 1), name="actor")
critic = Chain(Dense(64, 64, relu), Dense(64, 1), name="critic")
model = Chain(;
    feat_extr = Dense(4, 64, relu),
    actorCritic = Parallel(nothing;
        actor,
        critic
    )
)
model[2]

ps, st = Lux.setup(MersenneTwister(1234), model)
x = rand(Float32, 4, 1)
model(x, ps, st)
feats = model.feat_extr(x, ps.feat_extr, st.feat_extr)
actor_out = model.actorCritic.
critic_out = model.actorCritic.critic(feats)

ps
st