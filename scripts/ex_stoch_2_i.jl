# ==============================================================================
# Exercise 2.2 (i)
#
# This script:
#
#   1. Computes stochastic collocation solutions
#   2. Builds reconstructed density surfaces
#   3. Computes mean and variance
#   4. Compares ansatz spaces
#   5. Performs convergence studies
#
# ==============================================================================

include("../FV_stochastic.jl")

using Plots
using Statistics

# ==============================================================================
# PARAMETERS
# ==============================================================================

println()
println("====================================================")
println("Exercise 2.2 (i)")
println("====================================================")

# Spatial mesh
n = 400

# Test case
testcase = exercise_2_2_i

# Collocation levels for convergence study
M_values = [4, 8, 16, 32]

# Fine omega grid for visualization
omega_fine = collect(range(0.0, 1.0, length=200))

# Directory for figures
figdir = "figures/ex_stoch_2_i/"
isdir(figdir) || mkpath(figdir)

# ==============================================================================
# PART 1:STOCHASTIC SOLUTION FOR VISUALIZATION
# ==============================================================================

println()
println("Computing stochastic collocation solution...")

M_surface = 16

omegas, x_cells, solutions =
    stochastic_collocation_driver(
        n,
        testcase,
        M_surface;
        collocation_type = "lobatto"
    )

basis_surface = legendre_lobatto_basis(M_surface)

# ==============================================================================
# PART 2: DENSITY SURFACES
# ==============================================================================

println()
println("Building density surfaces...")

# ----------------------------------------------------------------------
# Constant reconstruction
# ----------------------------------------------------------------------

rho_const =
    build_density_surface(
        omega_fine,
        solutions,
        "constant"
    )

surface(
    omega_fine,
    x_cells,
    rho_const,
    xlabel = "ω",
    ylabel = "x",
    zlabel = "ρ(T,x,ω)",
    title = "Constant reconstruction"
)

savefig(figdir * "surface_constant.png")

# ----------------------------------------------------------------------
# Cubic spline reconstruction
# ----------------------------------------------------------------------

rho_cubic =
    build_density_surface(
        omega_fine,
        solutions,
        "cubic"
    )

surface(
    omega_fine,
    x_cells,
    rho_cubic,
    xlabel = "ω",
    ylabel = "x",
    zlabel = "ρ(T,x,ω)",
    title = "Cubic spline reconstruction"
)

savefig(figdir * "surface_cubic.png")

# ----------------------------------------------------------------------
# Polynomial reconstruction
# ----------------------------------------------------------------------

rho_poly =
    build_density_surface(
        omega_fine,
        solutions,
        "polynomial";
        basis_haupt = basis_surface
    )

surface(
    omega_fine,
    x_cells,
    rho_poly,
    xlabel = "ω",
    ylabel = "x",
    zlabel = "ρ(T,x,ω)",
    title = "Polynomial reconstruction"
)

savefig(figdir * "surface_polynomial.png")

println("Density surfaces saved.")

# ==============================================================================
# PART 3: MEAN AND VARIANCE
# ==============================================================================

println()
println("Computing mean and variance...")

mean_rho = vec(mean(rho_poly, dims=2))
var_rho  = vec(var(rho_poly, dims=2))

plot(
    x_cells,
    mean_rho,
    lw = 2,
    xlabel = "x",
    ylabel = "E[ρ]",
    title = "Mean density"
)

savefig(figdir * "mean_density.png")

plot(
    x_cells,
    var_rho,
    lw = 2,
    xlabel = "x",
    ylabel = "Var(ρ)",
    title = "Variance of density"
)

savefig(figdir * "variance_density.png")

println("Statistics saved.")

# ==============================================================================
# PART 4: COMPARE RECONSTRUCTIONS AT ONE OMEGA
# ==============================================================================

println()
println("Comparing ansatz spaces...")

omega_test = 0.37

sol_const =
    reconstruct_stochastic(
        omega_test,
        solutions,
        "constant"
    )

sol_cubic =
    reconstruct_stochastic(
        omega_test,
        solutions,
        "cubic"
    )

sol_poly =
    reconstruct_stochastic(
        omega_test,
        solutions,
        "polynomial";
        basis_haupt = basis_surface
    )

plot(
    x_cells,
    sol_const.rho,
    lw = 2,
    label = "constant",
    xlabel = "x",
    ylabel = "ρ"
)

plot!(
    x_cells,
    sol_cubic.rho,
    lw = 2,
    label = "cubic"
)

plot!(
    x_cells,
    sol_poly.rho,
    lw = 2,
    label = "polynomial"
)

title!("Reconstruction comparison (ω = $omega_test)")

savefig(figdir * "reconstruction_comparison.png")

println("Comparison plot saved.")

# ==============================================================================
# PART 5: CONVERGENCE STUDIES
# ==============================================================================

println()
println("Running convergence studies...")

# ----------------------------------------------------------------------
# Constant reconstruction
# ----------------------------------------------------------------------

conv_const =
    stochastic_convergence_study(
        n,
        testcase,
        M_values;
        reconstruction_method = "constant"
    )

# ----------------------------------------------------------------------
# Cubic reconstruction
# ----------------------------------------------------------------------

conv_cubic =
    stochastic_convergence_study(
        n,
        testcase,
        M_values;
        reconstruction_method = "cubic"
    )

# ----------------------------------------------------------------------
# Polynomial reconstruction
# ----------------------------------------------------------------------

conv_poly =
    stochastic_convergence_study(
        n,
        testcase,
        M_values;
        collocation_type = "lobatto",
        reconstruction_method = "polynomial"
    )

# ==============================================================================
# PART 6: CONVERGENCE PLOT
# ==============================================================================

plot(
    conv_const.M_values,
    conv_const.rho_L2,
    marker = :o,
    lw = 2,
    xscale = :log10,
    yscale = :log10,
    label = "constant",
    xlabel = "M",
    ylabel = "L² error"
)

plot!(
    conv_cubic.M_values,
    conv_cubic.rho_L2,
    marker = :o,
    lw = 2,
    label = "cubic"
)

plot!(
    conv_poly.M_values,
    conv_poly.rho_L2,
    marker = :o,
    lw = 2,
    label = "polynomial"
)

title!("Convergence study")

savefig(figdir * "convergence.png")

println("Convergence plot saved.")

# ==============================================================================
# PART 7: PRINT ERROR TABLE
# ==============================================================================

println()
println("====================================================")
println("Density L² errors")
println("====================================================")
println()

println("M\tconstant\t\tcubic\t\t\tpolynomial")

for k in eachindex(M_values)

    println(
        M_values[k], "\t",
        conv_const.rho_L2[k], "\t",
        conv_cubic.rho_L2[k], "\t",
        conv_poly.rho_L2[k]
    )

end

# ==============================================================================
# FINISHED
# ==============================================================================

println()
println("Exercise 2.2 (i) completed.")
println()
println("Generated figures:")
println("  " * figdir * "surface_constant.png")
println("  " * figdir * "surface_cubic.png")
println("  " * figdir * "surface_polynomial.png")
println("  " * figdir * "mean_density.png")
println("  " * figdir * "variance_density.png")
println("  " * figdir * "reconstruction_comparison.png")
println("  " * figdir * "convergence.png")
println()