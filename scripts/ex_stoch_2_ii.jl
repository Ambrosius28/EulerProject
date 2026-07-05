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

testcase = exercise_2_2_ii

parameters = Parameters(
    n = 100,
    M_values = [4, 8, 16],
    nomega_fine = 200,
    ansatz_space = "polynomial",
    nsnapshots = 4
)

# Directory for figures
figdir = "figures/ex_stoch_2_ii/"
isdir(figdir) || mkpath(figdir)

plot_heatmap_rho(testcase, parameters, figdir)
