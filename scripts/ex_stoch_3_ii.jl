# ==============================================================================
# Exercise 2.3 (ii)
# ==============================================================================

include("../FV_stochastic.jl")
include("../plots.jl")

# ==============================================================================
# PARAMETERS
# ==============================================================================

println()
println("====================================================")
println("Exercise 2.3 (ii)")
println("====================================================")

# Test case
testcase = exercise_2_3_ii

parameters = Parameters(
    n = 200,
    M_values = [64],
    omega_fine = collect(range(0.0, 1.0, length=200)),
    ansatz_space = "constant",
    nsnapshots = 4
)

# Directory for figures
figdir = "figures/ex_stoch_3_ii/"
isdir(figdir) || mkpath(figdir)

plot_mean_rho(testcase, parameters, figdir)