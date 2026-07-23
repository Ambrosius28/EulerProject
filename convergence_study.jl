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

"""
number of omegas --> omega_nodes, omega_weights (of the Gass-Legendre quadrature for ω ∈ (0,1))
"""
function gauss_legendre_omega(n_omegas::Int)
    nodes_ref, weights_ref = gausslegendre(n_omegas)   # on [-1, 1]
    omega_nodes   = 0.5 .* (nodes_ref .+ 1.0)          # on [0,1]
    omega_weights = 0.5 .* weights_ref                 # on [0,1] (for weight just the scaling counts)
    return omega_nodes, omega_weights
end

# ------------------------------------------------------------------------------
# Reference solution: M=64, method=cubic, t=T
# ------------------------------------------------------------------------------
"""
testcase::EulerTestCase, par_ref::Parameters --> U at final t as 3 index matrix: (component, x, ω)
"""
function reference_solution(testcase::EulerTestCase, par_ref::Parameters)
    sol_ref = main(testcase, par_ref)
    U = tensorize(sol_ref)
    return U[:, end, :, :] # 3 index matrix: (component, x, ω)
end

# ------------------------------------------------------------------------------
# Stochastic-collocation solution: t=T
# ------------------------------------------------------------------------------
function collocation_solution(testcase::EulerTestCase, par::Parameters)
    sol_fine = main(testcase, par)
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
# Error of the means in ω, for a single conserved variable component (1 = rho, 2 = m, 3 = E).
# dx: uniform spatial cell width (midpoint-rule weight for x, exact for FV
# piecewise-constant cell values).
# omega_weights: Gauss-Legendre quadrature weights on (0,1).
# ------------------------------------------------------------------------------
function means_error(
    U_M::Array{Float64,3},
    U_ref::Array{Float64,3},
    component::Int,
    p::Int, # we will use p = 2
    dx::Float64,
    omega_weights::Vector{Float64})

    # Mean in ω (L^1(Ω) norm)
    nx = size(U_M, 2)
    nomega = size(U_M, 3)

    U_M_mean = zeros(nx)
    U_ref_mean = zeros(nx)

    for i in 1:nx
        for k in 1:nomega
            U_M_mean[i] += omega_weights[k] * U_M[component, i, k]
            U_ref_mean[i] += omega_weights[k] * U_ref[component, i, k]
        end
    end

    # L^p([0,L]) error of the difference of the means is ω
    acc = 0.0
    @inbounds for i in eachindex(U_M_mean)
        acc += dx * abs(U_M_mean[i] - U_ref_mean[i])^p
    end

    return acc^(1 / p)
end

# Compute the convergence-in-measure error:
#   d(U_M,U_ref) = ∫_Ω min(1, ||U_M(·,ω)-U_ref(·,ω)||_{L¹([0,L])}) dP(ω)
#
# For each stochastic realization ω:
#   1. Compute the spatial L¹-error by summing over all finite volume cells.
#   2. Truncate the error with min(1, ·).
#   3. Integrate the truncated error over the stochastic space using the
#      quadrature weights omega_weights.
function measure_error(
    U_M::Array{Float64,3},
    U_ref::Array{Float64,3},
    component::Int,
    dx::Float64,
    omega_weights::Vector{Float64})

    nx = size(U_M, 2)
    nomega = size(U_M, 3)

    acc = 0.0
    l1_error = 0.0 
   
    for k in 1:nomega    # Loop over stochastic realizations

        for i in 1:nx    # Compute the spatial L¹ error for this ω
            l1_error += dx * abs(U_M[component, i, k] -
                                 U_ref[component, i, k])
        end

        # Integrate min(1, ||·||_{L¹}) over Ω
        acc += omega_weights[k] * min(1.0, l1_error)
    end

    return acc
end

# ------------------------------------------------------------------------------
# Run the convergence study for ONE test case and ONE ansatz method, over a
# sequence of M values. Returns M, L1/L2 errors and the EOC (Estimated/
# Empirical Order of Convergence) between successive M.
# ------------------------------------------------------------------------------
function convergence_study(
    testcase::EulerTestCase,
    par::Parameters,
    omega_weights::Vector{Float64},
    U_ref::Array{Float64,3};   # precomputed once per case, shared across methods
    component::Int = 1)

    dx = testcase.L / par.n
    method = par.ansatz_space

    errors_L1 = Float64[]
    errors_L2 = Float64[]
    errors_means = Float64[]
    errors_measure = Float64[]

    M_values = par.M_values
    println("  -> computing collocation solutions for M = $M_values, method = $method ...")
    for M in M_values
        par.M = M
        println("  -> method = $method, M = $M")
        U_M = collocation_solution(testcase, par)

        push!(errors_L1, Lp_error(U_M, U_ref, component, 1, dx, omega_weights))
        push!(errors_L2, Lp_error(U_M, U_ref, component, 2, dx, omega_weights))
        push!(errors_means, means_error(U_M, U_ref, component, 2, dx, omega_weights))
        push!(errors_measure, measure_error(U_M, U_ref, component, dx, omega_weights))
    end

    EOC_L1 = [NaN]
    EOC_L2 = [NaN]
    EOC_mean = [NaN]
    EOC_measure = [NaN]
    for i in 1:length(M_values)-1
        push!(EOC_L1, log2(errors_L1[i] / errors_L1[i+1]))
        push!(EOC_L2, log2(errors_L2[i] / errors_L2[i+1]))
        push!(EOC_mean, log2(errors_means[i] / errors_means[i+1]))
        push!(EOC_measure, log2(errors_measure[i] / errors_measure[i+1]))
    end

    return (M = M_values, 
            errors_L1 = errors_L1, errors_L2 = errors_L2, errors_means = errors_means, errors_measure = errors_measure,
            EOC_L1 = EOC_L1, EOC_L2 = EOC_L2, EOC_mean = EOC_mean, EOC_measure = EOC_measure)
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
    par::Parameters,
    omega_weights::Vector{Float64};
    component::Int = 1)

    dx = testcase.L / par.n
    method = par.ansatz_space

    M_values = par.M_values
    println("  -> (Cauchy) computing collocation solutions for M = $M_values, method = $method ...")
    
    U_list = Array{Float64,3}[]
    for M in M_values
        par.M = M
        push!(U_list, collocation_solution(testcase, par))
    end

    errors_L1 = Float64[]
    errors_L2 = Float64[]
    errors_means = Float64[]
    errors_measure = Float64[]
    for i in 1:length(M_values)-1
        push!(errors_L1, Lp_error(U_list[i], U_list[i+1], component, 1, dx, omega_weights))
        push!(errors_L2, Lp_error(U_list[i], U_list[i+1], component, 2, dx, omega_weights))
        push!(errors_means, means_error(U_list[i], U_list[i+1], component, 2, dx, omega_weights))
        push!(errors_measure, measure_error(U_list[i], U_list[i+1], component, dx, omega_weights))
    end

    # EOC between consecutive pairwise errors (needs at least 3 M-values to
    # get one EOC entry, same as the DGSEM mesh-convergence study).
    EOC_L1 = [NaN]
    EOC_L2 = [NaN]
    EOC_mean = [NaN]
    EOC_measure = [NaN]
    for i in 1:length(errors_L1)-1
        push!(EOC_L1, log2(errors_L1[i] / errors_L1[i+1]))
        push!(EOC_L2, log2(errors_L2[i] / errors_L2[i+1]))
        push!(EOC_mean, log2(errors_means[i] / errors_means[i+1]))
        push!(EOC_measure, log2(errors_measure[i] / errors_measure[i+1]))
    end

    # M reported here is the coarser of each compared pair (M_i vs M_{i+1}),
    # matching the convention "Fehler zwischen n=ns[i] und n=ns[i+1]" used
    # in the DGSEM convergence study.
    return (M = M_values[1:end-1],
            errors_L1 = errors_L1, errors_L2 = errors_L2, errors_means = errors_means, errors_measure = errors_measure,
            EOC_L1 = EOC_L1, EOC_L2 = EOC_L2, EOC_mean = EOC_mean, EOC_measure = EOC_measure)
end

function print_table(method::String, result)
    println()
    println("Method: $method")
    println("  " * "-"^132)

    @printf("  %-5s %-12s %-6s %-12s %-6s %-12s %-6s %-14s %-6s\n",
            "M",
            "L1", "EOC",
            "L2", "EOC",
            "Mean", "EOC",
            "Measure", "EOC")

    println("  " * "-"^132)

    for i in eachindex(result.M)
        @printf("  %-5d %-12.4e %-6.2f %-12.4e %-6.2f %-12.4e %-6.2f %-14.4e %-6.2f\n",
                result.M[i],
                result.errors_L1[i],      result.EOC_L1[i],
                result.errors_L2[i],      result.EOC_L2[i],
                result.errors_means[i],   result.EOC_mean[i],
                result.errors_measure[i], result.EOC_measure[i])
    end

    println("  " * "-"^132)
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
    par::Parameters,
    par_ref::Parameters;
    component::Int = 1,
    convergence_mode::Symbol = :reference,   # :reference or :cauchy
    figdir::String = "figures/convergence/")

    n = par.n
    n_gauss = par.nomega_fine
    M_values = par.M_values

    isdir(figdir) || mkpath(figdir)

    if convergence_mode == :reference && n_gauss <= maximum(M_values)
        @warn "n_gauss=$n_gauss is not much larger than max(M_values)=$(maximum(M_values)); " *
              "the omega-quadrature error may contaminate the measured convergence rate " *
              "(especially for the polynomial ansatz). Consider increasing n_gauss."
    end

    omega_eval, omega_weights = gauss_legendre_omega(n_gauss)
    title_base = get(CASE_TITLES, name, name)
    mode_label = get(MODE_LABELS, convergence_mode, String(convergence_mode))
    comp_name = component == 1 ? "ρ" : component == 2 ? "m" : "E"

    println()
    println("====================================================")
    println(title_base, "  (", mode_label, ")")
    println("====================================================")

    methods = ["constant", "cubic", "polynomial"]
    results = Dict{String,Any}()

    for method in methods
        par.ansatz_space = method

        result =
            if convergence_mode == :reference

                println("  -> building reference solution (n=$n, $(length(omega_eval)) Gauss-Legendre omega points)...")
                par_ref.ansatz_space = method
                U_ref = reference_solution(testcase, par_ref)

                convergence_study(testcase, par, omega_weights, U_ref;
                                  component = component)

            elseif convergence_mode == :cauchy

                cauchy_convergence_study(testcase, par, omega_weights;
                                         component = component)

            else
                error("Unknown convergence_mode: $convergence_mode (use :reference or :cauchy)")
            end

        results[method] = result
        print_table(method, result)
    end

    function legend_label(method::String, eoc_field::Symbol)
        eoc = getproperty(results[method], eoc_field)[end]
        eoc_str = isnan(eoc) ? "EOC: n/a" : @sprintf("EOC ≈ %.2f", eoc)
        return "$method ($eoc_str)"
    end

    subtitle = "n=$n cells, $(length(M_values)) collocation levels, component = $comp_name"

    error_symbol =
        convergence_mode == :reference ?
        "U_M - U_ref" :
        "U_{M+1} - U_M"

    mean_symbol =
        convergence_mode == :reference ?
        "𝔼[U_M] - 𝔼[U_ref]" :
        "𝔼[U_{M+1}] - 𝔼[U_M]"

    M_plotted = results["constant"].M
    xticks_setting = (M_plotted, string.(M_plotted))

    # -------------------- L1 convergence plot --------------------
    p1 = plot(
        xscale = :identity,
        yscale = :log10,
        xlabel = "M (number of collocation points)",
        xticks = xticks_setting,
        ylabel = "L¹ error  ‖$(error_symbol)‖ₗ₁(Ω×[0,L])",
        title = "$title_base\nL¹ convergence — $mode_label",
        titlefontsize = 10,
        legend = :bottomleft,
        legendfontsize = 8,
        grid = true,
        minorgrid = true,
    )

    for method in methods
        plot!(
            p1,
            results[method].M,
            results[method].errors_L1,
            marker = :circle,
            markersize = 5,
            linewidth = 2,
            label = legend_label(method, :EOC_L1),
        )
    end

    annotate!(p1, :bottomright, text(subtitle, 7, :gray, :right))

    savefig(
        p1,
        joinpath(
            figdir,
            "$(sanitize_filename(name))_$(convergence_mode)_L1_convergence.png",
        ),
    )

    # -------------------- L2 convergence plot --------------------
    p2 = plot(
        xscale = :identity,
        yscale = :log10,
        xlabel = "M (number of collocation points)",
        xticks = xticks_setting,
        ylabel = "L² error  ‖$(error_symbol)‖_{L¹(Ω×[0,L])}",
        title = "$title_base\nL² convergence — $mode_label",
        titlefontsize = 10,
        legend = :bottomleft,
        legendfontsize = 8,
        grid = true,
        minorgrid = true,
    )

    for method in methods
        plot!(
            p2,
            results[method].M,
            results[method].errors_L2,
            marker = :circle,
            markersize = 5,
            linewidth = 2,
            label = legend_label(method, :EOC_L2),
        )
    end

    annotate!(p2, :bottomright, text(subtitle, 7, :gray, :right))

    savefig(
        p2,
        joinpath(
            figdir,
            "$(sanitize_filename(name))_$(convergence_mode)_L2_convergence.png",
        ),
    )

    # -------------------- Mean convergence plot --------------------
    p3 = plot(
        xscale = :identity,
        yscale = :log10,
        xlabel = "M (number of collocation points)",
        xticks = xticks_setting,
        ylabel = "Mean error  ‖$(mean_symbol)‖ₗ₂([0,L])",
        title = "$title_base\nMean convergence — $mode_label",
        titlefontsize = 10,
        legend = :bottomleft,
        legendfontsize = 8,
        grid = true,
        minorgrid = true,
    )

    for method in methods
        plot!(
            p3,
            results[method].M,
            results[method].errors_means,
            marker = :circle,
            markersize = 5,
            linewidth = 2,
            label = legend_label(method, :EOC_mean),
        )
    end

    annotate!(p3, :bottomright, text(subtitle, 7, :gray, :right))

    savefig(
        p3,
        joinpath(
            figdir,
            "$(sanitize_filename(name))_$(convergence_mode)_mean_convergence.png",
        ),
    )

    # -------------------- Convergence-in-measure plot --------------------
    p4 = plot(
        xscale = :identity,
        yscale = :log10,
        xlabel = "M (number of collocation points)",
        xticks = xticks_setting,
        ylabel = "Convergence-in-measure error",
        title = "$title_base\nConvergence in measure — $mode_label",
        titlefontsize = 10,
        legend = :bottomleft,
        legendfontsize = 8,
        grid = true,
        minorgrid = true,
    )

    for method in methods
        plot!(
            p4,
            results[method].M,
            results[method].errors_measure,
            marker = :circle,
            markersize = 5,
            linewidth = 2,
            label = legend_label(method, :EOC_measure),
        )
    end

    annotate!(p4, :bottomright, text(subtitle, 7, :gray, :right))

    savefig(
        p4,
        joinpath(
            figdir,
            "$(sanitize_filename(name))_$(convergence_mode)_measure_convergence.png",
        ),
    )

    display(plot(
        p1, p2,
        p3, p4,
        layout = (2, 2),
        size = (1300, 900),
    ))

    return results
end


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

parameters = Parameters(
    n = 50,  
    M_values = [4, 8 , 12, 16, 24, 32],   
    nomega_fine = 40,  
)

parameters_ref = Parameters(
    n = parameters.n,
    M = 64,   # reference solution uses a much finer collocation
    nomega_fine = parameters.nomega_fine,
    # ansatz_space = "cubic",   # reference solution uses the most accurate ansatz
)

cases = [
    ("ex_2_2_i",   exercise_2_2_i),
    # ("ex_2_2_ii",  exercise_2_2_ii),
    # ("ex_2_3_i",   exercise_2_3_i),
    # ("ex_2_3_ii",  exercise_2_3_ii),
    # ("ex_2_4_i",   exercise_2_4_i),
    # ("ex_2_4_ii",  exercise_2_4_ii),
]

all_results = Dict{String, Any}()

for (name, testcase) in cases
    all_results["$(name)_reference"] = run_full_convergence_study(
        name, testcase, parameters, parameters_ref; convergence_mode = :reference)

    all_results["$(name)_cauchy"] = run_full_convergence_study(
        name, testcase, parameters, parameters_ref; convergence_mode = :cauchy)
end

println()
println("Convergence study complete (both :reference and :cauchy modes).")
println("Tables printed above; plots saved to figures/convergence/.")
