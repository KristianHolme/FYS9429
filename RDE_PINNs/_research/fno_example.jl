using NeuralOperators, Lux, Random, Optimisers, Zygote, CairoMakie

rng = Random.default_rng()

data_size = 128
m = 32

xrange = range(0, 2π; length=m) .|> Float32;
u_data = zeros(Float32, m, 1, data_size);
α = 0.5f0 .+ 0.5f0 .* rand(Float32, data_size);
v_data = zeros(Float32, m, 1, data_size);

for i in 1:data_size
    u_data[:, 1, i] .= sin.(α[i] .* xrange)
    v_data[:, 1, i] .= -inv(α[i]) .* cos.(α[i] .* xrange)
end

fno = FourierNeuralOperator(gelu; chs=(1, 64, 64, 128, 1), modes=(16,), permuted=Val(true))

ps, st = Lux.setup(rng, fno);
data = [(u_data, v_data)];

function train!(model, ps, st, data; epochs=10)
    losses = []
    tstate = Training.TrainState(model, ps, st, OptimizationOptimisers.Adam(0.01f0))
    for _ in 1:epochs, (x, y) in data
        _, loss, _, tstate = Training.single_train_step!(AutoZygote(), MSELoss(), (x, y),
            tstate)
        push!(losses, loss)
    end
    return losses
end

losses = train!(fno, ps, st, data; epochs=100)
##
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Epoch", ylabel="Loss", xscale=log10, yscale=log10)
lines!(ax, losses)
fig