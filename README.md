# Euler Project

Team project for "Mathematical Modelling Lab" course at the Johannes Gutenburg University of Mainz. 

Solver for compressible inviscid Euler equations in 1D.

Authors: Areej Ghabayen, Jan-Eric Fries, Marco Ambrogio Bergamo, Sara Jalaouy.


## Usage Guide — FV + Explicit Euler + Stochastic Collocation Solver

---

## Overview

The main entry point for running a full simulation is `main()`. It solves the 1D compressible Euler equations using:

- **Space**: Finite Volume method with Local Lax-Friedrichs (Rusanov) flux
- **Time**: Explicit Euler with adaptive CFL time stepping (common dt across all collocation nodes)
- **Stochastic**: Stochastic collocation in the random parameter `ω ∈ [0, 1]`, with reconstruction on a fine `ω` grid

---

## Running a Simulation

```julia
fine_stoch = main(n, M, testcase, omega_fine, reconstruction_method)
```

### Inputs

| Argument | Type | Description |
|---|---|---|
| `n` | `Int` | Number of spatial cells |
| `M` | `Int` | Number of stochastic collocation nodes in `ω` |
| `testcase` | `EulerTestCase` | Struct defining the problem (see below) |
| `omega_fine` | `Vector{Float64}` | Fine grid of `ω` values for reconstruction (e.g. 200 points in `[0,1]`) |
| `reconstruction_method` | `String` | One of `"constant"`, `"cubic"`, `"polynomial"` |
| `cfl_parameter` | `Float64` | *(keyword, default `0.8`)* CFL number for adaptive time stepping |

### Reconstruction methods and collocation nodes

| `reconstruction_method` | Node type used internally |
|---|---|
| `"constant"` | Uniform nodes |
| `"cubic"` | Uniform nodes |
| `"polynomial"` | Gauss-Lobatto-Legendre nodes |

### Predefined test cases

The following `EulerTestCase` instances are defined and ready to use:

| Name | Description |
|---|---|
| `exercise_2_2_i` | Smooth IC, periodic BC, `γ = 1.2`, discontinuity location random in `ω` |
| `exercise_2_2_ii` | Sod shock tube, Neumann BC, `γ = 1.4`, shock location random in `ω` |
| `exercise_2_3_i` | Smooth IC, periodic BC, `γ(ω) = 1.1 + 0.5ω` random |
| `exercise_2_3_ii` | Sod shock tube, Neumann BC, `γ(ω) = 1.1 + 0.5ω` random |
| `exercise_2_4_i` | Supersonic inflow, custom BC, time- and `ω`-dependent inflow density and pressure |
| `exercise_2_4_ii` | Supersonic inflow, custom BC, `ω`-dependent pressure only |

### Example call

```julia
omega_fine = collect(range(0.0, 1.0, length = 200))

fine_stoch = main(
    100,                # n: 100 spatial cells
    10,                 # M: 10 collocation nodes
    exercise_2_3_i,     # test case
    omega_fine,         # fine omega grid
    "cubic";            # reconstruction method
    cfl_parameter = 0.8
)
```

---

## Return Type

`main()` returns a `StochasticSolution`:

```julia
struct StochasticSolution
    omegas::Vector{Float64}                      # fine omega grid, length K
    solutions::Vector{DeterministicSolution}     # one entry per omega, length K
end
```

Each `DeterministicSolution` contains:

```julia
struct DeterministicSolution
    times::Vector{Float64}       # time steps [t_0, t_1, ..., t_end], length nt
    U::Vector{Matrix{Float64}}   # solution at each time step, length nt
                                 # each matrix is (3, n): rows = [ρ, m, E], cols = spatial cells
end
```

> **Note:** Ghost cells are **not** stored. Each `U[j]` has size `(3, n)`, where rows are the three
> conservative variables `ρ` (density), `m` (momentum), `E` (energy), and columns are the `n` interior spatial cells.

---

## Accessing the Data

### Time steps

```julia
times = fine_stoch.solutions[1].times   # Vector of length nt, same for all ω
nt    = length(times)
```

### Fine omega grid

```julia
omega_fine = fine_stoch.omegas          # Vector of length K
K = length(omega_fine)
```

### Solution at a specific time step `j` and omega index `k`

```julia
U = fine_stoch.solutions[k].U[j]       # Matrix (3, n)

rho = U[1, :]   # density,   length n
m   = U[2, :]   # momentum,  length n
E   = U[3, :]   # energy,    length n
```

### Cell centers (x grid)

```julia
dx = testcase.L / n
x  = [(i - 0.5) * dx for i in 1:n]    # length n
```

---

## Building Plots

### Density profile at final time for a single ω

```julia
k = 1                                  # index into omega_fine
U_final = fine_stoch.solutions[k].U[end]
plot(x, U_final[1, :], xlabel="x", ylabel="ρ", label="ω=$(omega_fine[k])")
```

### Density surface ρ(x, ω) at time step j

```julia
j = nt                                 # e.g. final time
Z = zeros(K, n)
for (k, sol) in enumerate(fine_stoch.solutions)
    Z[k, :] .= sol.U[j][1, :]
end
heatmap(x, omega_fine, Z, xlabel="x", ylabel="ω", title="ρ(x,ω), t=$(times[j])")
```

### Mean and variance over ω at final time

```julia
rho_matrix = hcat([sol.U[end][1, :] for sol in fine_stoch.solutions]...)  # (n, K)

mean_rho = mean(rho_matrix, dims=2)    # mean over ω, shape (n, 1)
var_rho  = var(rho_matrix,  dims=2)    # variance over ω, shape (n, 1)

plot(x, mean_rho, ribbon=sqrt.(var_rho), label="mean ± std")
```

### Animation over time

```julia
anim = @animate for j in 1:nt
    Z = zeros(K, n)
    for (k, sol) in enumerate(fine_stoch.solutions)
        Z[k, :] .= sol.U[j][1, :]
    end
    heatmap(x, omega_fine, Z, xlabel="x", ylabel="ω", title="t=$(round(times[j], digits=4))")
end
gif(anim, "rho.gif", fps=6)
```

---

## Tensorized Access

For convenience, `tensorize()` reshapes the full solution into a 4D array:

```julia
U_tensor = tensorize(fine_stoch)        # size (3, nt, n, K)
                                        # dims: (component, time, space, omega)

rho_surface = U_tensor[1, end, :, :]   # density at final time, shape (n, K)
```
