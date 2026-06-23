# ==============================================================================
# Exercise 2.2 (ii)
# ==============================================================================

include("../FV_stochastic.jl")
include("../plots.jl")

# ==============================================================================
# PARAMETERS
# ==============================================================================

println()
println("====================================================")
println("Exercise 2.2 (ii)")
println("====================================================")

# Spatial mesh
n = 400

# Test case
testcase = exercise_2_2_ii

# Collocation levels for convergence study
M_values = [4, 8, 16, 32]

# Fine omega grid for visualization
omega_fine = collect(range(0.0, 1.0, length=200))

# Directory for figures
figdir = "figures/ex_stoch_2_ii/"
isdir(figdir) || mkpath(figdir)

plot_heatmap_rho(testcase,
    M_values,
    "constant",
    n,
    4,
    omega_fine,
    figdir)