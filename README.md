# Euler Project

Team project for "Mathematical Modelling Lab" course at the Johannes Gutenberg University of Mainz.

Solver for compressible inviscid Euler equations in 1D.

**Authors:** Areej Ghabayen, Jan-Eric Fries, Marco Ambrogio Bergamo, Sara Jalaouy.

**Always start the Julia REPL with**

```julia -t auto --project=.```

(for enabling multi-thread and packages needed).

---

## Overview

The solver tackles the 1D compressible Euler equations using:

- **Space**: Finite Volume method with Local Lax-Friedrichs (Rusanov) flux
- **Time**: Explicit Euler with adaptive CFL time stepping
- **Uncertainty quantification**: Stochastic collocation over a random parameter `ω ∈ [0, 1]`, with reconstruction on a fine `ω` grid

---

## File structure

```
.
├── FV_stochastic.jl   # solver: types, FV update, stochastic collocation, reconstruction
├── plots.jl           # plotting functions (heatmaps, mean ± std, animations)
└── scripts/
    ├── ex_2_2_i.jl
    ├── ex_2_2_ii.jl
    └── ...
```

---

## Quick start
**Always start the Julia REPL with**

```julia -t auto --project=.```

(for enabling multi-thread and packages needed).

Each exercise script follows the same pattern:

```julia
include("../FV_stochastic.jl")
include("../plots.jl")

testcase = exercise_2_2_i

parameters = Parameters(
    n          = 200,
    M          = 8,                                    # collocation nodes for a single run
    M_values   = [4, 8, 16],                           # loop over M for convergence plots
    omega_fine = collect(range(0.0, 1.0, length=200)),
    ansatz_space = "cubic",
    nsnapshots = 10,
)

figdir = "figures/ex_2_2_i/"
plot_heatmap_rho(testcase, parameters, figdir)
```

---

## The `Parameters` struct

All numerical and stochastic settings are collected in a single `Parameters` struct defined with `Base.@kwdef`, so every field has a name and a default value.

```julia
Base.@kwdef struct Parameters
    n             :: Int                   # number of spatial cells
    M             :: Int       = 3         # collocation nodes — used by main() for a single run
    M_values      :: Vector{Int} = [3]     # list of M values — used by plotting functions
    omega_fine    :: Vector{Float64}       # fine ω grid for reconstruction
    ansatz_space  :: String                # reconstruction method (see below)
    cfl_parameter :: Float64   = 0.8      # CFL number for adaptive time stepping
    nsnapshots    :: Int       = 20        # number of time snapshots to save
end
```

### `M` vs `M_values` — which one to use

These two fields serve different purposes and are used in different contexts:

| Field | Used by | Purpose |
|---|---|---|
| `M` | `main(testcase, par)` | Single run with a fixed number of collocation nodes |
| `M_values` | `plot_heatmap_rho`, `plot_mean_rho` | Loop over multiple values of M to compare results |

**Single run** — set `M` and call `main` directly:

```julia
par = Parameters(
    n = 100, M = 10,
    omega_fine = collect(range(0.0, 1.0, length=200)),
    ansatz_space = "cubic",
)
fine_stoch = main(testcase, par)
```

**Convergence/comparison plots** — set `M_values` and call a plot function, which internally rebuilds a `Parameters` with each `M` in the list:

```julia
par = Parameters(
    n = 200, M_values = [4, 8, 16],
    omega_fine = collect(range(0.0, 1.0, length=200)),
    ansatz_space = "cubic",
    nsnapshots = 6,
)
plot_heatmap_rho(testcase, par, "figures/convergence/")
```

> **Note:** when using only plotting functions you can leave `M` at its default value of `3`; it will not be read. Conversely, when calling `main` directly, `M_values` is ignored.

### `ansatz_space` and collocation nodes

The reconstruction method also determines which collocation nodes are used internally:

| `ansatz_space` | Node type | Description |
|---|---|---|
| `"constant"` | Uniform | Piecewise constant reconstruction in ω |
| `"cubic"` | Uniform | Cubic spline reconstruction in ω |
| `"polynomial"` | Gauss-Lobatto-Legendre | Global polynomial (spectral) reconstruction in ω |

---

## Predefined test cases

| Name | BC | γ | Randomness |
|---|---|---|---|
| `exercise_2_2_i` | Periodic | 1.2 | Discontinuity location ∝ ω |
| `exercise_2_2_ii` | Neumann | 1.4 | Shock location ∝ ω (Sod tube) |
| `exercise_2_3_i` | Periodic | 1.1 + 0.5ω | γ random |
| `exercise_2_3_ii` | Neumann | 1.1 + 0.5ω | γ random (Sod tube) |
| `exercise_2_4_i` | Custom (inflow) | 1.4 | Time- and ω-dependent inflow ρ and p |
| `exercise_2_4_ii` | Custom (inflow) | 1.4 | ω-dependent inflow p only |

---

## Running a simulation with `main`

```julia
fine_stoch = main(testcase, par)
```

`main` runs the three-stage pipeline:

1. **Stochastic collocation driver** — solves the deterministic FV problem for each of the `M` collocation nodes in parallel (via `Threads.@threads`).
2. **Reconstruction** — interpolates the node solutions onto the fine `ω` grid using the chosen `ansatz_space`.

### Return type

`main` returns a `StochasticSolution`:

```julia
struct StochasticSolution
    omegas    :: Vector{Float64}               # fine ω grid, length K
    solutions :: Vector{DeterministicSolution} # one per ω value, length K
end
```

Each `DeterministicSolution` contains:

```julia
struct DeterministicSolution
    times :: Vector{Float64}       # saved time points [t_0, …, T], length nt
    U     :: Vector{Matrix{Float64}} # solution at each time, length nt
                                   # each matrix is (3, n): rows = [ρ, m, E]
end
```

> Ghost cells are **not** stored. Each `U[j]` has size `(3, n)`.

---

## Accessing the solution

### Time steps and ω grid

```julia
times     = fine_stoch.solutions[1].times   # Vector of length nt
omega_fine = fine_stoch.omegas              # Vector of length K
```

### Solution at time step `j` and ω index `k`

```julia
U   = fine_stoch.solutions[k].U[j]   # Matrix (3, n)
rho = U[1, :]                         # density
m   = U[2, :]                         # momentum
E   = U[3, :]                         # energy
```

### Cell centres

```julia
dx = testcase.L / par.n
x  = [(i - 0.5) * dx for i in 1:par.n]
```

### 4D tensor (component × time × space × ω)

```julia
U_tensor    = tensorize(fine_stoch)     # size (3, nt, n, K)
rho_surface = U_tensor[1, end, :, :]   # density at final time, shape (n, K)
```

---

## Plotting functions

Both functions loop over `par.M_values`, run a full simulation for each `M`, and save results into `figdir`.

### `plot_heatmap_rho(testcase, par, figdir)`

Produces:
- PNG snapshots of the density heatmap ρ(x, ω) at selected times
- An animated GIF over all time steps

Output files follow the pattern `rho_M{M}_{ansatz_space}.gif` and `rho_M{M}_snap{j}.png`.

### `plot_mean_rho(testcase, par, figdir)`

Produces:
- PNG snapshots of the mean density E[ρ](x) ± std as a ribbon plot
- An animated GIF

Output files follow the pattern `mean_rho_M{M}_{ansatz_space}.gif`.

### Example

```julia
parameters = Parameters(
    n            = 200,
    M_values     = [8, 16],
    omega_fine   = collect(range(0.0, 1.0, length=200)),
    ansatz_space = "cubic",
    nsnapshots   = 4,
)

plot_heatmap_rho(exercise_2_2_i, parameters, "figures/ex_2_2_i/")
plot_mean_rho(exercise_2_2_i,   parameters, "figures/ex_2_2_i/")
```

---

## Dependencies

```julia
using LinearAlgebra, Printf, Statistics
using Polynomials
using Plots
using Trixi       # for Gauss-Lobatto-Legendre nodes and interpolation
using Dierckx     # for cubic spline reconstruction
using Base.Threads
```

Install with:

```julia
using Pkg
Pkg.add(["Polynomials", "Plots", "Trixi", "Dierckx"])
```
