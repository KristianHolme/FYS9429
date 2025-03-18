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

@kwdef struct TrainingHistory
    losses::Vector{Float32} = []
    epochs::Vector{Int} = []
    learning_rates::Vector{Float32} = []
end

@kwdef struct FNOConfig
    chs::NTuple = (3, 64, 64, 64, 2)
    modes::Int = 16
    activation::Function = gelu
    history::TrainingHistory = TrainingHistory()
end

function FNO(config::FNOConfig)
    return FourierNeuralOperator(config.activation; chs=config.chs, modes=(config.modes,), permuted=Val(true))
end

