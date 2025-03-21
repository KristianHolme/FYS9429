function train_and_save!(config::FNOConfig, data, lr::AbstractArray{<:Real}, epochs::AbstractArray{<:Int}; 
        folder="",
        kwargs...)
    train!(config, data, lr, epochs; kwargs...)
    plot_losses(config; saveplot=true)
    safesave(datadir("fno", folder, savename(config, "jld2")), config)
    return config
end



    