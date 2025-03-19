@kwdef struct TrainingHistory
    losses::Vector{Float32} = []
    epochs::Vector{Int} = []
    learning_rates::Vector{Float32} = []
end

function Base.show(io::IO, history::TrainingHistory)
    println(io, "TrainingHistory(")
    println(io, "  losses: $(length(history.losses)) values ")
    if !isempty(history.losses)
        println(io, "         min: $(minimum(history.losses))")
        println(io, "         max: $(maximum(history.losses))")
        println(io, "         mean: $(mean(history.losses))")
        println(io, "         latest: $(history.losses[end])")
    end
    println(io, "  epochs=$(history.epochs),")
    println(io, "  learning_rates=$(history.learning_rates)")
    print(io, ")")
end

mutable struct FNOConfig
    chs::NTuple{N,Int} where N
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

Base.show(io::IO, config::FNOConfig) = print(io, "FNOConfig(chs=$(config.chs), modes=$(config.modes), activation=$(config.activation))")

function FNO(config::FNOConfig)
    return FourierNeuralOperator(config.activation; chs=config.chs, modes=(config.modes,), permuted=Val(true))
end
function FNO(;chs=(3, 64, 64, 64, 2), modes=16, activation=gelu)
    return FourierNeuralOperator(activation; chs=chs, modes=(modes,), permuted=Val(true))
end


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

function train!(config::FNOConfig, data; lr::AbstractFloat=3e-4, epochs::Int=10, dev=cpu_device(), AD=AutoZygote())
    config.ps = config.ps |> dev
    config.st = config.st |> dev
    model = FNO(config)
    hist = config.history
    train!(model, config.ps, config.st, data; losses=hist.losses, lr, epochs, dev, AD)
    push!(hist.learning_rates, lr)
    push!(hist.epochs, epochs)
    return nothing
end

function train!(config::FNOConfig, data;
     lr::AbstractArray{<:Real}=3e-4,
     epochs::AbstractArray{<:Int}=10,
     dev=cpu_device(),
     AD=AutoZygote()
    )
    @assert length(lr) == length(epochs) "lr and epochs must have the same length"
    for (lr, epochs) in zip(lr, epochs)
        train!(config, data; lr, epochs, dev, AD)
    end
end


