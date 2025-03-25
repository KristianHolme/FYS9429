function train_and_save!(config::FNOConfig, data, lr::AbstractArray{<:Real}, epochs::AbstractArray{<:Int}; 
        folder="",
        kwargs...)
    train!(config, data, lr, epochs; kwargs...)
    plot_losses(config; saveplot=true)
    safesave(datadir("fno", folder, savename(config, "jld2")), config)
    return config
end



function moving_average(x, window)
    y = zeros(length(x))
    for i in 1:length(x)
        if i < window
            y[i] = mean(x[1:i])
        else
            y[i] = mean(x[i-window+1:i])
        end
    end
    return y
end