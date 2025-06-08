using DrWatson
@quickactivate :project_2
using Lux
using DRiL
using Zygote
using WGLMakie
WGLMakie.activate!()
using Statistics
using LinearAlgebra
using ClassicControlEnvironments
##

ps = rand(1:4, 2, 2, 3) .|> Float32
log_std = Float32[0.1 0.2 ; 0.3 0.4]

actions = rand(Float32,2,2,3)

function myloss(ps)
    ds = DiagGaussian.(eachslice(ps, dims=ndims(ps)), Ref(log_std))
    return mean(logpdf.(ds, eachslice(actions, dims=ndims(actions))))
end

Zygote.gradient(myloss, ps)