using DRL_RDE_utils
using ProgressMeter
using NeuralOperators
using Lux
using Optimisers
using Zygote
using LinearAlgebra
using Random
using OptimizationOptimisers
##
RDEParams = RDEParam(tmax = 400.0f0)
env = RDEEnv(RDEParams,
    dt = 0.5f0,
    observation_strategy=SectionedStateObservation(minisections=32))

policies = [ConstantRDEPolicy(env), 
 ScaledPolicy(SinusoidalRDEPolicy(env,w_2=0.5f0), 0.5f0),
 ScaledPolicy(RandomRDEPolicy(env), 0.5f0)
]
data = []
for policy in policies, _ in 1:10
    @info "Collecting data for $(policy)"
    sim_data = run_policy(policy, env)
    n_data = length(sim_data.observations)

    raw_data = zeros(Float32, 32, 2, n_data)
    x_data = @view raw_data[:,:,1:end-1]
    y_data = @view raw_data[:,:,2:end]

    for i in eachindex(sim_data.observations)
        obs = sim_data.observations[i]
        raw_data[:, 1, i] = obs[1:32]
        raw_data[:, 2, i] = obs[33:64]
    end
    push!(data, (x_data, y_data))
end

fno = FourierNeuralOperator(gelu; chs=(2, 64, 64, 128, 2), modes=(16,), permuted=Val(true))

rng = Random.default_rng()
ps, st = Lux.setup(rng, fno);

function train!(model, ps, st, data; epochs=10)
    losses = []
    tstate = Training.TrainState(model, ps, st, OptimizationOptimisers.Adam(0.01f0))
    p = Progress(epochs*length(data))
    for _ in 1:epochs, (x, y) in data
        _, loss, _, tstate = Training.single_train_step!(AutoZygote(), MSELoss(), (x, y),
            tstate)
        push!(losses, loss)
        next!(p, showvalues=[("Loss", loss)])
    end
    return losses
end

losses = train!(fno, ps, st, data; epochs=100)
##
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Epoch", ylabel="Loss", xscale=log10, yscale=log10)
lines!(ax, losses)
fig
## Test the model
RDEParams = RDEParam(tmax = 2.0)
env = RDEEnv(RDEParams,
    dt = 0.01,
    observation_strategy=SectionedStateObservation(minisections=32))
sim_test_data = run_policy(ConstantRDEPolicy(env), env)
test_observations = sim_test_data.observations[100:end]

test_data = zeros(Float32, 32, 2, length(test_observations))
for i in eachindex(test_observations)
    obs = test_observations[i]
    test_data[:, 1, i] = obs[1:32]
    test_data[:, 2, i] = obs[33:64]
end

output_data, st = Lux.apply(fno, test_data[:,:,1:end-1], ps, st)
##
fig = Figure()
ax_u = Axis(fig[1, 1], xlabel="Time", ylabel="u")
ax_λ = Axis(fig[1, 2], xlabel="Time", ylabel="λ")
ls = []
for i in [1, 50, 100]
    u_true = lines!(ax_u, test_data[:, 1, i+1], color=:red, linestyle=:dash)
    u_pred = lines!(ax_u, output_data[:, 1, i], color=:blue)
    λ_true = lines!(ax_λ, test_data[:, 2, i+1], color=:red, linestyle=:dash)
    λ_pred = lines!(ax_λ, output_data[:, 2, i], color=:blue)
    if i == 1
        push!(ls, u_true)
        push!(ls, u_pred)
        push!(ls, λ_true)
        push!(ls, λ_pred)
    end
end
Legend(fig[2,:], ls, ["True u", "Predicted u", "True λ", "Predicted λ"], vertical=false, tellheight=false)
fig
