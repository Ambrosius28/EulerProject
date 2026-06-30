# ==============================================================================
# Convergence study (in omega) for the stochastic collocation Euler solver
# ==============================================================================
# IMPORTANT — kept fixed across an M-sweep for a given test case:
#   - spatial resolution n
#   - cfl_parameter
# Only M (number of collocation points) is varied, so that the measured
# error isolates the omega-discretization error and is not contaminated by
# spatial discretization error.
# ==============================================================================

include("./FV_stochastic.jl")

using Printf, Statistics, Plots
using FastGaussQuadrature   # trustworthy, lightweight package for Gauss nodes/weights. (Pkg.add("FastGaussQuadrature") if not yet installed)

# ------------------------------------------------------------------------------
# Gauss-Legendre nodes/weights on (0,1) (mapped from the standard [-1,1]
# rule). Used to integrate the error in omega essentially exactly (see note
# above), instead of a naive trapezoidal rule on a fixed fine grid.
# ------------------------------------------------------------------------------

"number of omegas --> omega_nodes, omega_weights (of the Gass-Legendre quadrature for ω ∈ (0,1))"
function gauss_legendre_omega(n_omegas::Int)
    nodes_ref, weights_ref = gausslegendre(n_omegas)   # on [-1, 1]
    omega_nodes   = 0.5 .* (nodes_ref .+ 1.0)          # on [0,1]
    omega_weights = 0.5 .* weights_ref                 # on [0,1] (for weight just the scaling counts)
    return omega_nodes, omega_weights
end

# ------------------------------------------------------------------------------
# Reference solution: M=64, method=cubic, t=T
# ------------------------------------------------------------------------------
function reference_solution(n, testcase, omega_fine; cfl_parameter=0.8, M_ref=64, ref_method="polynomial")
    sol_ref = main(n, M_ref, testcase, omega_fine, ref_method; cfl_parameter=cfl_parameter)
    U = tensorize(sol_ref)
    return U[:, end, :, :]
end

# ------------------------------------------------------------------------------
# Stochastic-collocation solution: t=T
# ------------------------------------------------------------------------------
function collocation_solution(n, M, testcase, omega_fine, method; cfl_parameter=0.8)
    sol_fine = main(n, M, testcase, omega_fine, method; cfl_parameter = cfl_parameter)
    U = tensorize(sol_fine)
    return U[:, end, :, :]
end

# ------------------------------------------------------------------------------
# L^p error, computed jointly over x and omega, for a single conserved
# variable component (1 = rho, 2 = m, 3 = E).
# dx: uniform spatial cell width (midpoint-rule weight for x, exact for FV
# piecewise-constant cell values).
# omega_weights: Gauss-Legendre quadrature weights on (0,1).
# ------------------------------------------------------------------------------
function Lp_error(
    U_M::Array{Float64,3},
    U_ref::Array{Float64,3},
    component::Int,
    p::Int,
    dx::Float64,
    omega_weights::Vector{Float64})

    diff = U_M[component, :, :] .- U_ref[component, :, :]   # (x, omega)
    nx, nomega = size(diff)

    acc = 0.0
    for k in 1:nomega
        for i in 1:nx
            acc += dx * omega_weights[k] * abs(diff[i, k])^p
        end
    end

    return acc^(1.0 / p)
end

# ------------------------------------------------------------------------------
# Run the convergence study for ONE test case and ONE ansatz method, over a
# sequence of M values. Returns M, L1/L2 errors and the EOC (Estimated/
# Empirical Order of Convergence) between successive M.
# ------------------------------------------------------------------------------
function convergence_study(
    testcase::EulerTestCase,
    n::Int,
    M_values::Vector{Int},
    method::String,
    omega_eval::Vector{Float64},
    omega_weights::Vector{Float64},
    U_ref::Array{Float64,3};   # precomputed once per case, shared across methods
    component::Int = 1,
    cfl_parameter::Float64 = 0.8)

    dx = testcase.L / n

    errors_L1 = Float64[]
    errors_L2 = Float64[]

    for M in M_values
        println("  -> method = $method, M = $M")
        U_M = collocation_solution(n, M, testcase, omega_eval,  method; cfl_parameter = cfl_parameter)

        push!(errors_L1, Lp_error(U_M, U_ref, component, 1, dx, omega_weights))
        push!(errors_L2, Lp_error(U_M, U_ref, component, 2, dx, omega_weights))
    end

    EOC_L1 = [NaN]
    EOC_L2 = [NaN]
    for i in 1:length(M_values)-1
        push!(EOC_L1, log2(errors_L1[i] / errors_L1[i+1]))
        push!(EOC_L2, log2(errors_L2[i] / errors_L2[i+1]))
    end

    return (M = M_values, errors_L1 = errors_L1, errors_L2 = errors_L2,
            EOC_L1 = EOC_L1, EOC_L2 = EOC_L2)
end

# ------------------------------------------------------------------------------
# CAUCHY-SEQUENCE convergence study (alternative to the reference-based one
# above): compares consecutive collocation solutions U_{M_i} and U_{M_{i+1}}
# directly to each other instead of to an independently computed reference.
#
# This mirrors how the deterministic convergence study (Part 1, mesh
# refinement) was done: comparing successive refinement levels to each
# other rather than to a separately computed "exact" solution. If the
# sequence of errors e(M_i) = ||U_{M_i} - U_{M_{i+1}}|| shrinks as M grows,
# this is evidence that {U_M} is a Cauchy sequence, hence converges to some
# limit even without independently knowing/computing that limit.
#
# Cheaper than the reference-based study too: no extra deterministic solves
# at n_gauss points are needed -- it reuses only the M-collocation runs.
# ------------------------------------------------------------------------------
function cauchy_convergence_study(
    testcase::EulerTestCase,
    n::Int,
    M_values::Vector{Int},
    method::String,
    omega_eval::Vector{Float64},
    omega_weights::Vector{Float64};
    component::Int = 1,
    cfl_parameter::Float64 = 0.8)

    dx = testcase.L / n

    println("  -> (Cauchy) computing collocation solutions for M = $M_values, method = $method ...")
    U_list = [collocation_solution(n, M, testcase, omega_eval,  method; cfl_parameter = cfl_parameter)
              for M in M_values]

    errors_L1 = Float64[]
    errors_L2 = Float64[]
    for i in 1:length(M_values)-1
        push!(errors_L1, Lp_error(U_list[i], U_list[i+1], component, 1, dx, omega_weights))
        push!(errors_L2, Lp_error(U_list[i], U_list[i+1], component, 2, dx, omega_weights))
    end

    # EOC between consecutive pairwise errors (needs at least 3 M-values to
    # get one EOC entry, same as the DGSEM mesh-convergence study).
    EOC_L1 = [NaN]
    EOC_L2 = [NaN]
    for i in 1:length(errors_L1)-1
        push!(EOC_L1, log2(errors_L1[i] / errors_L1[i+1]))
        push!(EOC_L2, log2(errors_L2[i] / errors_L2[i+1]))
    end

    # M reported here is the coarser of each compared pair (M_i vs M_{i+1}),
    # matching the convention "Fehler zwischen n=ns[i] und n=ns[i+1]" used
    # in the DGSEM convergence study.
    return (M = M_values[1:end-1], errors_L1 = errors_L1, errors_L2 = errors_L2,
            EOC_L1 = EOC_L1, EOC_L2 = EOC_L2)
end


function print_table(method::String, result)
    println()
    println("  Method: $method")
    println("  " * "-"^60)
    @printf("  %-6s %-14s %-10s %-14s %-10s\n", "M", "L1 error", "EOC", "L2 error", "EOC")
    for i in 1:length(result.M)
        @printf("  %-6d %-14.6e %-10.3f %-14.6e %-10.3f\n",
            result.M[i], result.errors_L1[i], result.EOC_L1[i],
            result.errors_L2[i], result.EOC_L2[i])
    end
    println("  " * "-"^60)
end

# ------------------------------------------------------------------------------
# Human-readable titles for each test case (for plots/presentation), instead
# of the internal short names like "ex_2_2_i".
# ------------------------------------------------------------------------------
const CASE_TITLES = Dict(
    "ex_2_2_i"  => "Random Initial Data — Periodic BC (γ=1.2)",
    "ex_2_2_ii" => "Random Initial Data — Neumann BC, Shock Tube (γ=1.4)",
    "ex_2_3_i"  => "Random γ(ω) = 1.1 + 0.5ω — Periodic BC",
    "ex_2_3_ii" => "Random γ(ω) = 1.1 + 0.5ω — Neumann BC, Shock Tube",
    "ex_2_4_i"  => "Random Boundary Data (Case i)",
    "ex_2_4_ii" => "Random Boundary Data (Case ii)",
)

const MODE_LABELS = Dict(
    :reference => "vs. reference solution",
    :cauchy    => "Cauchy / self-convergence",
)

# ------------------------------------------------------------------------------
# Turn a string into a filesystem-safe filename (letters/digits/underscore only).
# ------------------------------------------------------------------------------
sanitize_filename(s::String) = replace(s, r"[^A-Za-z0-9]+" => "_")

# ------------------------------------------------------------------------------
# Run the full convergence study (all 3 ansatz spaces) for ONE test case,
# print tables, and produce L1/L2 log-lin comparison plots.
# ------------------------------------------------------------------------------
function run_full_convergence_study(
    name::String,
    testcase::EulerTestCase,
    n::Int,
    M_values::Vector{Int};
    component::Int = 1,
    cfl_parameter::Float64 = 0.8,
    n_gauss::Int = 60,
    convergence_mode::Symbol = :reference,   # :reference or :cauchy
    figdir::String = "figures/convergence/")

    isdir(figdir) || mkpath(figdir)

    if convergence_mode == :reference && n_gauss <= maximum(M_values)
        @warn "n_gauss=$n_gauss is not much larger than max(M_values)=$(maximum(M_values)); " *
              "the omega-quadrature error may contaminate the measured convergence rate " *
              "(especially for the polynomial ansatz). Consider increasing n_gauss."
    end

    omega_eval, omega_weights = gauss_legendre_omega(n_gauss)
    title_base  = get(CASE_TITLES, name, name)
    mode_label  = get(MODE_LABELS, convergence_mode, String(convergence_mode))
    comp_name   = component == 1 ? "ρ" : component == 2 ? "m" : "E"

    println()
    println("====================================================")
    println(title_base, "  (", mode_label, ")")
    println("====================================================")

    # Compute the reference solution ONCE per case (it doesn't depend on the
    # ansatz method), and reuse it for all 3 methods below -- avoids solving
    # the same n_gauss deterministic problems 3 times over.
    U_ref = if convergence_mode == :reference
        println("  -> building reference solution (n=$n, $(length(omega_eval)) Gauss-Legendre omega points)...")
        reference_solution(n, testcase, omega_eval; cfl_parameter = cfl_parameter)
    else
        nothing
    end

    methods = ["constant", "cubic", "polynomial"]
    results = Dict{String, Any}()

    for method in methods
        result = if convergence_mode == :reference
            convergence_study(testcase, n, M_values, method, omega_eval, omega_weights, U_ref;
                               component = component, cfl_parameter = cfl_parameter)
        elseif convergence_mode == :cauchy
            cauchy_convergence_study(testcase, n, M_values, method, omega_eval, omega_weights;
                                      component = component, cfl_parameter = cfl_parameter)
        else
            error("Unknown convergence_mode: $convergence_mode (use :reference or :cauchy)")
        end
        results[method] = result
        print_table(method, result)
    end

    # Build a legend label per method that includes the last computed EOC,
    # so the key convergence-rate number is visible directly on the plot
    # without needing to consult the printed table separately.
    function legend_label(method::String, eoc_field::Symbol)
        eoc = getproperty(results[method], eoc_field)[end]
        eoc_str = isnan(eoc) ? "EOC: n/a" : @sprintf("EOC ≈ %.2f", eoc)
        return "$method ($eoc_str)"
    end

    subtitle = "n=$n cells, $(length(M_values)) collocation levels, component = $comp_name"

    # -------------------- L1 convergence plot --------------------
    M_plotted = results["constant"].M
    xticks_setting = (M_plotted, string.(M_plotted))

    p1 = plot(xscale = :identity, yscale = :log10,
              xlabel = "M (number of collocation points)",
              xticks = xticks_setting,
              ylabel = "L¹ error  ‖U_M − U_ref‖_{L¹(Ω×[0,L])}",
              title = "$title_base\nL¹ convergence — $mode_label",
              titlefontsize = 10, subplot = 1,
              legend = :bottomleft, legendfontsize = 8,
              grid = true, minorgrid = true)
    for method in methods
        plot!(p1, results[method].M, results[method].errors_L1,
              marker = :circle, markersize = 5, linewidth = 2,
              label = legend_label(method, :EOC_L1))
    end
    annotate!(p1, :bottomright, text(subtitle, 7, :gray, :right))
    savefig(p1, joinpath(figdir, "$(sanitize_filename(name))_$(convergence_mode)_L1_convergence.png"))

    # -------------------- L2 convergence plot --------------------
    p2 = plot(xscale = :identity, yscale = :log10,
              xlabel = "M (number of collocation points)",
              xticks = xticks_setting,
              ylabel = "L² error  ‖U_M − U_ref‖_{L²(Ω×[0,L])}",
              title = "$title_base\nL² convergence — $mode_label",
              titlefontsize = 10,
              legend = :bottomleft, legendfontsize = 8,
              grid = true, minorgrid = true)
    for method in methods
        plot!(p2, results[method].M, results[method].errors_L2,
              marker = :circle, markersize = 5, linewidth = 2,
              label = legend_label(method, :EOC_L2))
    end
    annotate!(p2, :bottomright, text(subtitle, 7, :gray, :right))
    savefig(p2, joinpath(figdir, "$(sanitize_filename(name))_$(convergence_mode)_L2_convergence.png"))

    display(plot(p1, p2, layout = (1, 2), size = (1200, 480)))

    return results
end

#
# n and M_values can be tuned per case; keep n FIXED across the M-sweep of a
# given case so only the omega-discretization error is being probed.
# n_gauss (default 60) is the number of Gauss-Legendre points used to
# integrate the error in omega for the :reference mode; keep it well above
# max(M_values) so the omega-quadrature error stays negligible compared to
# the collocation error. (n_gauss is unused for :cauchy, but still controls
# the common evaluation grid for the error norm there too.)
#
# Both modes are run for every case:
#   :reference -> absolute error vs. an independently computed solution
#   :cauchy    -> self-convergence (consecutive M's compared to each other),
#                 consistent with how the deterministic mesh-refinement
#                 study in Part 1 was done.

cases = [
    #("ex_2_2_i",   exercise_2_2_i,   50, [4, 8, 12, 16, 24, 32], 40), #n, M, number of omega_fine
    ("ex_2_2_ii",  exercise_2_2_ii,  50, [4, 8, 12, 16, 24, 32], 40),
    ("ex_2_3_i",   exercise_2_3_i,   50, [4, 8, 12, 16, 24, 32], 40),
    ("ex_2_3_ii",  exercise_2_3_ii,  50, [4, 8, 12, 16, 24, 32], 40),
    ("ex_2_4_i",   exercise_2_4_i,   50, [4, 8, 12, 16, 24, 32], 40),
    ("ex_2_4_ii",  exercise_2_4_ii,  50, [4, 8, 12, 16, 24, 32], 40),
]

all_results = Dict{String, Any}()

for (name, testcase, n, M_values, n_gauss) in cases
    all_results["$(name)_reference"] = run_full_convergence_study(
        name, testcase, n, M_values; n_gauss = n_gauss, convergence_mode = :reference)

    all_results["$(name)_cauchy"] = run_full_convergence_study(
        name, testcase, n, M_values; n_gauss = n_gauss, convergence_mode = :cauchy)
end

println()
println("Convergence study complete (both :reference and :cauchy modes).")
println("Tables printed above; plots saved to figures/convergence/.")