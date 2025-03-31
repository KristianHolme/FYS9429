# RDEML: Machine Learning for Rotating Detonation Engines

RDEML is a Julia package that provides machine learning tools for project 1 in FYS9429. It started using PINNs to solve the RDE equations (see [RDE.jl](https://github.com/KristianHolme/RDE.jl)), but has since pivoted to focus on examining the use of Fourier Neural Operators (FNOs) for solving the RDE equations.

Dependecies that are not registered julia packages are:
- [RDE.jl](https://github.com/KristianHolme/RDE.jl) for simulator for RDE equations
- [RDE_Env.jl](https://github.com/KristianHolme/RDE_Env.jl) for RL environmets, policies etc.
- [RLBridge](https://github.com/KristianHolme/RLBridge.jl) for connecting the julia environments to the stable-baselines3 and sbx (stable-baselines-jax) python packages for training the agents used in the data generation.

For other dependencies that are registered julia packages, see [Project.toml](Project.toml)