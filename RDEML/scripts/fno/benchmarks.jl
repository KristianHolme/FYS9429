using DrWatson
@quickactivate :RDEML
using BenchmarkTools
using IntervalSets
"""
On nam-shub-02
Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz
NVIDIA A10
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
BenchmarkTools.Trial: 2146 samples with 1 evaluation per sample.
 Range (min … max):  1.206 ms … 57.006 ms  ┊ GC (min … max): 0.00% … 92.46%
 Time  (median):     1.678 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   2.315 ms ±  2.602 ms  ┊ GC (mean ± σ):  8.73% ±  8.82%

  ██▆▅▄▃▃▂▂                                                  ▁
  ████████████▇▇▇▆▆▅▆▆▆▅▅▅▄▆▄▄▄▅▅▅▃▄▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄ █
  1.21 ms      Histogram: log(frequency) by time     17.3 ms <

 Memory estimate: 731.52 KiB, allocs estimate: 13173.
"""
##
df = collect_results(datadir("fno", "final_runs_2"))
idx = argmin(df.final_test_loss)
rng = Random.default_rng()
_, test_data = prepare_dataset(;rng, batch_size=1, test_split=0.2, create_loader=false)
x_data = test_data.raw_x_data
N = size(x_data, 3)
fno_config = df[idx, :].full_config
model = FNO(fno_config)
const cdev = cpu_device()
const gdev = CUDADevice(device!(0))
ps, st = fno_config.ps, fno_config.st

function eval_fno(model, ps, st, input, gdev, cdev; move_to_gdev=true, move_back=true)
    if move_to_gdev
        input = input |> gdev
    end
    input = reshape(input, 512, 3, 1)
    y_pred, _ = Lux.apply(model, input, ps, st)
    if move_back
        y_pred = y_pred |> cdev
    end
    return y_pred
end

## benchmark fno on gpu
ps = ps |> gdev
st = st |> gdev
#precompile
@time eval_fno(model, ps, st, x_data[:,:,1], gdev, cdev)

@benchmark eval_fno(model, ps, st, input, gdev, cdev) setup=(input = x_data[:,:,rand(1:N)])
"""
BenchmarkTools.Trial: 3137 samples with 1 evaluation per sample.
 Range (min … max):  889.458 μs … 9.046 ms  ┊ GC (min … max):  0.00% … 82.34%
 Time  (median):     971.067 μs             ┊ GC (median):     0.00%
 Time  (mean ± σ):     1.576 ms ± 1.652 ms  ┊ GC (mean ± σ):  34.19% ± 24.27%

  █▆                                                    ▂▃▂▁   
  ██▄▃▃██▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▄██████ █
  889 μs       Histogram: log(frequency) by time      6.31 ms <

 Memory estimate: 83.75 KiB, allocs estimate: 2409.
 """
# without moving to and from gpu
x_data_gdev = x_data |> gdev
@time eval_fno(model, ps, st, x_data_gdev[:,:,1], gdev, cdev, move_to_gdev=false, move_back=false);
@benchmark eval_fno(model, ps, st, input, gdev, cdev, move_to_gdev=false, move_back=false) setup=(input = x_data_gdev[:,:,rand(1:N)])
"""
BenchmarkTools.Trial: 3231 samples with 1 evaluation per sample.
 Range (min … max):  839.197 μs … 8.442 ms  ┊ GC (min … max):  0.00% … 81.35%
 Time  (median):     927.808 μs             ┊ GC (median):     0.00%
 Time  (mean ± σ):     1.506 ms ± 1.605 ms  ┊ GC (mean ± σ):  34.58% ± 24.20%

  █▇                                                    ▁▃▃▂▁ ▁
  ██▃▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃█████ █
  839 μs       Histogram: log(frequency) by time      6.08 ms <

 Memory estimate: 78.64 KiB, allocs estimate: 2385.
"""
## benchmark fno on cpu
ps = ps |> cdev
st = st |> cdev
@benchmark eval_fno(model, ps, st, input, cdev,cdev, move_to_gdev=false, move_back=false) setup=(input = x_data[:,:,rand(1:N)])
"""
BenchmarkTools.Trial: 1611 samples with 1 evaluation per sample.
 Range (min … max):  1.829 ms … 226.613 ms  ┊ GC (min … max):  0.00% …  4.31%
 Time  (median):     2.024 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   3.079 ms ±   6.331 ms  ┊ GC (mean ± σ):  28.76% ± 25.60%

  ▇█▆▃                                        ▂▁▁▁▁▁           
  ████▇▅▅▄▅▅▄▁▁▁▁▁▁▁▁▁▁▁▄▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▄▇█████████▇▅▅▅▄▆▁▄ █
  1.83 ms      Histogram: log(frequency) by time      8.71 ms <

 Memory estimate: 25.11 MiB, allocs estimate: 365.
"""