# Minimal script to debug runtime dispatch in Zygote pullbacks through
# DRiL.DRiLDistributions.SquashedDiagGaussian logpdf.

using Random
# using Zygote
using DRiL
using DRiL.DRiLDistributions
using Lux
using Cthulhu
# using InteractiveUtils
using JET
##
Random.seed!(42)

call_back(back, y) = back(y)

T = Float32

const dim = 3
const mean = rand(T, dim)
const log_std = fill(T(-0.1), dim)
# pick an x in (-1, 1)
x = tanh.(mean)

const d = SquashedDiagGaussian(mean, log_std)

# Sanity
lp = logpdf(d, x)
println("logpdf(d, x) => ", lp, " :: ", typeof(lp))

# 1) Pullback wrt mean
f_mean(x) = logpdf(SquashedDiagGaussian(mean, log_std), x)
f_meanD(x) = logpdf(DiagGaussian(mean, log_std), x)

@code_warntype f_mean(x)
@code_warntype f_meanD(x)

@report_opt f_mean(x)
@report_opt f_meanD(x)
