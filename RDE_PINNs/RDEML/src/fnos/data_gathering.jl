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

function make_data_reset_strategies()
    return [ShiftReset(NShock(1)),
    ShiftReset(NShock(2)),
    ShiftReset(NShock(3)),
    ShiftReset(NShock(4)),
    ShiftReset(SineCombination(;modes=2:9)),
    ShiftReset(RandomCombination()),
    ShiftReset(RandomCombination()),
    ShiftReset(RandomCombination()),
    ]
end

function collect_data(policies, envs, reset_strategies; n_runs_per_reset_strategy=2)
    n_runs = length(reset_strategies)*n_runs_per_reset_strategy
    run_data = Vector{Any}(undef, length(policies)*n_runs)
    data = Vector{Tuple{Array{Float32, 3}, Array{Float32, 3}}}(undef, length(policies)*n_runs)
    prog = Progress(n_runs*length(policies), "Collecting data...")
    data_collect_stats = zeros(Int, length(policies), n_runs)
    for policy_i in eachindex(policies)
        for reset_strategy_i in eachindex(reset_strategies)
            for run_i in 1:n_runs_per_reset_strategy
                i = (policy_i-1)*n_runs + (reset_strategy_i-1)*n_runs_per_reset_strategy + run_i
                policy = policies[policy_i]
                env = envs[policy_i]
                N = env.prob.params.N
                reset_strategy = reset_strategies[reset_strategy_i]
                env.prob.reset_strategy = reset_strategy
                sim_data = run_policy(policy, env)
                run_data[i] = sim_data

                if env.terminated
                    @info "Policy $(policies[policy_i]) with reset strategy
                         $(reset_strategies[reset_strategy_i]) crashed at run $run_i"
                end
                
                n_data = length(sim_data.states)
                raw_data = zeros(Float32, N, 3, n_data)
                x_data = @view raw_data[:,:,1:end-1]
                y_data = @view raw_data[:,1:2,2:end]
                
                for j in eachindex(sim_data.observations)
                    obs = sim_data.states[j]
                    raw_data[:, 1, j] = obs[1:N]
                    raw_data[:, 2, j] = obs[N+1:2N]
                    raw_data[:, 3, j] .= sim_data.u_ps[j]
                end
                data[i] = (x_data, y_data)
                data_collect_stats[policy_i, (reset_strategy_i-1)*n_runs_per_reset_strategy + run_i] += 1
                next!(prog, showvalues=[("$(policies[policy_i])", sum(data_collect_stats[policy_i,:])) for policy_i in eachindex(policies)])
            end
        end
    end
    @info "Done collecting data, collected $(length(policies)) policies 
        × $(length(reset_strategies)) reset strategies × $n_runs_per_reset_strategy 
        runs = $(length(policies)*length(reset_strategies)*n_runs_per_reset_strategy) 
        total runs"
    return run_data, data
end

function save_data(run_data, data; filename="data.jld2")
    jldsave(datadir(filename); run_data, data)
end




