# ==============================================================================
# Exercise 2.2 (i)
# ==============================================================================

include("../FV_stochastic.jl")
include("../plots.jl")

# ==============================================================================
# PARAMETERS
# ==============================================================================

println()
println("====================================================")
println("Exercise 2.2 (i)")
println("====================================================")

testcase = exercise_2_2_i

parameters = Parameters(
    n = 200,
    M_values = [8, 16],
    omega_fine = collect(range(0.0, 1.0, length=200)),
    ansatz_space = "cubic",
    nsnapshots = 4
)

# Directory for figures
figdir = "figures/ex_stoch_2_i/"
isdir(figdir) || mkpath(figdir)

plot_heatmap_rho(testcase, parameters, figdir)