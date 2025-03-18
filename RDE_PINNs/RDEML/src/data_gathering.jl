make_env(target_shock_count=1) = RDEEnv(RDEParam(tmax = 400.0f0),
    dt = 1.0f0,
    observation_strategy=SectionedStateObservation(minisections=32, target_shock_count=target_shock_count),
    reset_strategy=ShiftReset(RandomShockOrCombination())
)

function make_data_policies_and_envs()
    envs = [make_env() for _ in 1:7]
    for i in 1:4
        push!(envs, make_env(i))
    end
    N = envs[1].prob.params.N
    policies = [ConstantRDEPolicy(envs[1]), 
    ScaledPolicy(SinusoidalRDEPolicy(envs[2], w_1=0f0, w_2=0.5f0), 0.5f0),
    ScaledPolicy(RandomRDEPolicy(envs[3]), 0.5f0),
    ScaledPolicy(RandomRDEPolicy(envs[4]), 0.2f0),
    ScaledPolicy(RandomRDEPolicy(envs[5]), 0.1f0),
    ScaledPolicy(RandomRDEPolicy(envs[6]), 0.05f0),
    StepwiseRDEPolicy(envs[7], [20.0f0, 100.0f0, 200.0f0, 350.0f0], 
            [0.64f0, 0.86f0, 0.64f0, 0.96f0])
    ]
    for i in 1:4
        push!(policies, load_best_policy("transition_rl_9", project_path=joinpath(homedir(), "Code", "DRL_RDE"),
        filter_fn=df -> df.target_shock_count == i)[1])
    end
    return policies, envs
end



