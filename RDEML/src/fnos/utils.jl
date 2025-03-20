function train_and_save!(config::FNOConfig, data, lr::AbstractArray{<:Real}, epochs::AbstractArray{<:Int})
    train!(config, data, lr, epochs)
    plot_losses(config; saveplot=true)
    safesave(datadir("fno", savename(config, "jld2")), config)
    return config
end

    