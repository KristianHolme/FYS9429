using DrWatson
@quickactivate :RDEML
using BenchmarkTools
"""
On nam-shub-01
Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GH
NVIDIA RTX 2080 Ti
"""

##
struct NoReward <: RDE_Env.AbstractRDEReward end
function RDE_Env.set_reward!(env::RDE_Env.AbstractRDEEnv, rt::NoReward)
    env.reward = 0f0
end

params = RDEParam(tmax=1000f0)
env = RDEEnv(params, dt=1f0, action_type=ScalarPressureAction(), 
    reward_type=NoReward(),
    reset_strategy=RandomCombination())
reset!(env)
@benchmark act!(env, action) setup=(action = rand(-0.05f0..0.05f0)) teardown=(env.done && reset!(env))
"""
BenchmarkTools.Trial: 976 samples with 1 evaluation per sample.
 Range (min … max):  2.358 ms … 22.576 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     3.979 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   5.103 ms ±  3.152 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █ ▃                                                         
  █▄█▄▄▄▄▅▄▄▄▄▄▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▂▃▂▂▂▂▃▂▂▂▂▁▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂ ▃
  2.36 ms        Histogram: frequency by time        15.6 ms <

 Memory estimate: 731.42 KiB, allocs estimate: 13170.
"""
##
df = collect_results(datadir("fno", "final_runs_2"))
idx = argmin(df.final_test_loss)
_, test_data = prepare_dataset(;rng, batch_size=1, test_split=0.2, create_loader=false)
x_data = test_data.raw_x_data
N = size(x_data, 3)
fno_config = df[idx, :].full_config
model = FNO(fno_config)
const cdev = cpu_device()
const gdev = CUDADevice(device!(0))
ps, st = fno_config.ps, fno_config.st

function eval_fno(model, ps, st, input, dev)
    input = input |> dev
    input = reshape(input, 512, 3, 1)
    y_pred, _ = Lux.apply(model, input, ps, st)
    return y_pred
end

## benchmark fno on gpu
ps = ps |> gdev
st = st |> gdev
@time eval_fno(model, ps, st, x_data[:,:,1], gdev)

@benchmark eval_fno(model, ps, st, input, gdev) setup=(input = x_data[:,:,rand(1:N)])
"""
BenchmarkTools.Trial: 3418 samples with 1 evaluation per sample.
 Range (min … max):  686.855 μs … 15.791 ms  ┊ GC (min … max):  0.00% … 85.32%
 Time  (median):     718.528 μs              ┊ GC (median):     0.00%
 Time  (mean ± σ):     1.441 ms ±  2.850 ms  ┊ GC (mean ± σ):  43.16% ± 19.42%

  █                                                        ▁    
  ██▁▃▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅▅▇▇▇███ █
  687 μs        Histogram: log(frequency) by time      13.3 ms <

 Memory estimate: 50.20 KiB, allocs estimate: 1531.
 """

## benchmark fno on cpu
ps = ps |> cdev
st = st |> cdev
@benchmark eval_fno(model, ps, st, input, cdev) setup=(input = x_data[:,:,rand(1:N)])
"""
BenchmarkTools.Trial: 850 samples with 1 evaluation per sample.
 Range (min … max):  4.195 ms … 28.837 ms  ┊ GC (min … max):  0.00% … 82.96%
 Time  (median):     4.684 ms              ┊ GC (median):     0.00%
 Time  (mean ± σ):   5.839 ms ±  3.042 ms  ┊ GC (mean ± σ):  19.35% ± 21.53%

   █▁                                                         
  ███▆██▅▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▂▃▃▃▃▃▃▂ ▃
  4.2 ms         Histogram: frequency by time          14 ms <

 Memory estimate: 34.05 MiB, allocs estimate: 276.
"""