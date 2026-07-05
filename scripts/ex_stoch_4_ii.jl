# ==============================================================================
# Exercise 2.4 (ii)
# ==============================================================================

include("../FV_stochastic.jl")
include("../plots.jl")

# ==============================================================================
# PARAMETERS
# ==============================================================================

println()
println("====================================================")
println("Exercise 2.4 (ii)")
println("====================================================")

# Test case
testcase = exercise_2_4_ii

parameters = Parameters(
    n = 100,
    M_values = [64],
    nomega_fine = 200,
    ansatz_space = "constant",
    nsnapshots = 4
)

# Directory for figures
figdir = "figures/ex_stoch_4_ii/"
isdir(figdir) || mkpath(figdir)

plot_mean_rho(testcase, parameters, figdir)