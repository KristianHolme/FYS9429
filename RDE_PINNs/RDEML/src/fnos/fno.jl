function train!(model, ps, st, data; lr = 3f-4, epochs=10, losses=[], dev=cpu_device(), AD=AutoZygote())
    tstate = Training.TrainState(model, ps, st, OptimizationOptimisers.Adam(lr))
    p = Progress(epochs*length(data))
    for _ in 1:epochs, (x, y) in dev(data)
        _, loss, _, tstate = Training.single_train_step!(AD, MSELoss(), (x, y),
        tstate)
        push!(losses, loss)
        next!(p, showvalues=[("Loss", loss)])
    end
    return losses
end

function train!(config::FNOConfig, data; lr=3e-4, epochs=10, dev=cpu_device(), kwargs...)
    config.ps = config.ps |> dev
    config.st = config.st |> dev
    model = FNO(config)
    hist = config.history
    train!(model, config.ps, config.st, data; losses=hist.losses, lr, epochs, dev, kwargs...)
    push!(hist.learning_rates, lr)
    push!(hist.epochs, epochs)
    return config
end

@kwdef struct TrainingHistory
    losses::Vector{Float32} = []
    epochs::Vector{Int} = []
    learning_rates::Vector{Float32} = []
end

mutable struct FNOConfig
    chs::NTuple
    modes::Int
    activation::Function
    ps::NamedTuple
    st::NamedTuple
    history::TrainingHistory
    function FNOConfig(;chs=(3, 64, 64, 64, 2), modes=16, activation=gelu, rng=Random.default_rng())
        ps, st = Lux.setup(rng, FNO(;chs, modes, activation))
        return new(chs, modes, activation, ps, st, TrainingHistory())
    end
end

function FNO(config::FNOConfig)
    return FourierNeuralOperator(config.activation; chs=config.chs, modes=(config.modes,), permuted=Val(true))
end
function FNO(;chs=(3, 64, 64, 64, 2), modes=16, activation=gelu)
    return FourierNeuralOperator(activation; chs=chs, modes=(modes,), permuted=Val(true))
end

