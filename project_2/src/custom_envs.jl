using DRiL
using Random

"""
    GoalReachingEnv

A simple 1D goal-reaching environment where:
- The goal is a random point in (-1, 1)
- Observation space contains [player_position, goal_position] 
- Action space is in (-1, 1) with linear interpolation movement
- Reward is 1 - (distance_to_goal / 2), keeping it in (0, 1)
- Actions are mapped linearly: 0=no movement, 1=move to position 1, -1=move to position -1, 0.5=move halfway to 1
"""
mutable struct GoalReachingEnv <: AbstractEnv
    player_position::Float32
    goal_position::Float32
    observation_space::UniformBox{Float32}
    action_space::UniformBox{Float32}
    step_count::Int
    max_steps::Int
    _terminated::Bool
    _truncated::Bool
    _last_reward::Float32
    _info::Dict{String,Any}
    rng::Random.AbstractRNG

    function GoalReachingEnv(max_steps::Int=200, rng::Random.AbstractRNG=Random.Xoshiro())
        # Observation: [player_position, goal_position], both in (-1, 1)
        obs_space = UniformBox{Float32}(-1.0f0, 1.0f0, (2,))
        # Action: continuous in (-1, 1)
        act_space = UniformBox{Float32}(-1.0f0, 1.0f0, (1,))
        
        new(0.0f0, 0.0f0, obs_space, act_space, 0, max_steps, false, false, 0.0f0, Dict{String,Any}(), rng)
    end
end

# Implement required DRiL interface methods
DRiL.observation_space(env::GoalReachingEnv) = env.observation_space
DRiL.action_space(env::GoalReachingEnv) = env.action_space
DRiL.terminated(env::GoalReachingEnv) = env._terminated
DRiL.truncated(env::GoalReachingEnv) = env._truncated
DRiL.get_info(env::GoalReachingEnv) = env._info

function DRiL.reset!(env::GoalReachingEnv)
    # Reset episode state
    env.step_count = 0
    env._terminated = false
    env._truncated = false
    env._last_reward = 0.0f0
    env._info = Dict{String,Any}()
    
    # Set random goal position in (-1, 1)
    env.goal_position = rand(env.rng, Float32) * 2.0f0 - 1.0f0
    
    # Set initial player position (could be random or fixed)
    env.player_position = rand(env.rng, Float32) * 2.0f0 - 1.0f0
    
    return DRiL.observe(env)
end

function DRiL.observe(env::GoalReachingEnv)
    return Float32[env.player_position, env.goal_position]
end

function DRiL.act!(env::GoalReachingEnv, action::AbstractArray)
    return DRiL.act!(env, action[1])
end

function DRiL.act!(env::GoalReachingEnv, action::Float32)
    env.step_count += 1
    
    # Linear interpolation movement mapping
    if action >= 0.0f0
        # Move towards position 1: new_pos = current + action * (1 - current)
        env.player_position = env.player_position + action * (1.0f0 - env.player_position)
    else
        # Move towards position -1: new_pos = current + |action| * (-1 - current)
        env.player_position = env.player_position + (-action) * (-1.0f0 - env.player_position)
    end
    
    # Clamp to bounds to handle floating point precision issues
    env.player_position = clamp(env.player_position, -1.0f0, 1.0f0)
    
    # Calculate reward: 1 - (distance / 2) to keep it in (0, 1)
    distance = abs(env.player_position - env.goal_position)
    reward = 1.0f0 - (distance / 2.0f0)
    
    # Check termination conditions
    env._truncated = env.step_count >= env.max_steps
    # Could add early termination if very close to goal
    env._terminated = false  # This env doesn't naturally terminate
    
    env._last_reward = reward
    env._info = Dict{String,Any}(
        "step" => env.step_count,
        "distance_to_goal" => distance,
        "player_position" => env.player_position,
        "goal_position" => env.goal_position
    )
    
    return reward
end