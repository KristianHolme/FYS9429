@kwdef mutable struct PendulumProblem
    theta::Float32 = rand(Float32) * 2π
    velocity::Float32 = (rand(Float32) * 16.0f0 - 8.0f0)
    torque::Float32 = 0.0f0
    mass::Float32 = 1.0f0
    length::Float32 = 1.0f0
    gravity::Float32 = 9.81f0
    dt::Float32 = 0.01f0
end

mutable struct PendulumEnv <: AbstractEnv
    problem::PendulumProblem
    action_space::UniformBox
    observation_space::UniformBox
    step::Int
    function PendulumEnv(;problem = PendulumProblem(rand(Float32) * 2π, (rand(Float32) * 16.0f0 - 8.0f0),
        0.0f0, 1.0f0, 1.0f0, 9.81f0, 0.01f0))
        
        action_space = UniformBox(Float32, -2.0f0, 2.0f0, (1,))
        observation_space = UniformBox(Float32, -1f0, 1f0, (3,))
        env = new(problem, action_space, observation_space, 0)
        return env
    end
end

function reset!(env::PendulumEnv)
    reset!(env.problem)
    env.step = 0
end

function reset!(problem::PendulumProblem)
    problem.theta = rand(Float32) * 2π
    problem.velocity = (rand(Float32) * 16.0f0 - 8.0f0)
    problem.torque = 0.0f0
end

function act!(env::PendulumEnv, action::Float32)
    pend = env.problem
    pend.torque = action
    g = pend.gravity
    m = pend.mass
    L = pend.length
    dt = pend.dt
    theta = pend.theta
    pend.velocity += ( (3*g/2L)*sin(theta) + 3/(m*L^2)*pend.torque ) * dt
    pend.theta += pend.velocity * dt
    env.step += 1
end

function observe(env::PendulumEnv)
    x = cos(env.problem.theta)
    y = sin(env.problem.theta)
    scaled_vel = env.problem.velocity / 8.0f0
    return [x, y, scaled_vel]
end

terminated(env::PendulumEnv) = false
truncated(env::PendulumEnv) = env.step >= 200
action_space(env::PendulumEnv) = env.action_space
observation_space(env::PendulumEnv) = env.observation_space
get_info(env::PendulumEnv) = Dict(:step => env.step)

function reward(env::PendulumEnv)
    theta = env.problem.theta
    velocity = env.problem.velocity
    reward = -(theta^2 + 0.1f0*velocity^2 + 0.001f0*env.problem.torque^2)
    return reward
end
