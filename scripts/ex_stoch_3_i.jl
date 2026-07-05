# ==============================================================================
# Exercise 2.3 (i)
# ==============================================================================

include("../FV_stochastic.jl")
include("../plots.jl")

# ==============================================================================
# PARAMETERS
# ==============================================================================

println()
println("====================================================")
println("Exercise 2.3 (i)")
println("====================================================")

# Test case
testcase = exercise_2_3_i

parameters = Parameters(
    n = 200,
    M_values = [16],
    nomega_fine = 200,
    ansatz_space = "constant",
    nsnapshots = 20
)

# Directory for figures
figdir = "figures/ex_stoch_3_i/"
isdir(figdir) || mkpath(figdir)

plot_mean_rho(testcase, parameters, figdir)