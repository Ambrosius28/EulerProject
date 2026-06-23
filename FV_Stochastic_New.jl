# ==============================================================================
# 1D Compressible Euler Equations with Random Data
# Finite Volume (Rusanov/LLF) + Explicit Euler in time
# + Stochastic Collocation in omega (piecewise-constant / cubic spline / polynomial)
#
# SINGLE SCRIPT VERSION.
# Dependencies: Plots, Dierckx (only). Install with:
#   import Pkg; Pkg.add(["Plots", "Dierckx"])
# ==============================================================================

using LinearAlgebra, Printf, Statistics
using Plots
using Dierckx   # cubic spline reconstruction in omega

# ------------------------------------------------------------------------------
# Output directory for figures
# ------------------------------------------------------------------------------
const FIGDIR = "figures"
isdir(FIGDIR) || mkpath(FIGDIR)

# ==============================================================================
# SECTION 1 — Euler flux, wave speed, LLF numerical flux
# ==============================================================================

function physical_flux(U::AbstractVector, gamma::Float64)
    rho, m, E = U[1], U[2], U[3]
    u = m / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * u^2)
    return [m, rho * u^2 + p, (E + p) * u]
end

function primitive_to_conservative(rho::Float64, u::Float64, p::Float64, gamma::Float64)
    m = rho * u
    E = p / (gamma - 1.0) + 0.5 * rho * u^2
    return [rho, m, E]
end

function max_wave_speed(U_left::AbstractVector, U_right::AbstractVector, gamma::Float64)
    eps_safe = 1.0e-12
    rho_L, m_L, E_L = U_left[1],  U_left[2],  U_left[3]
    rho_R, m_R, E_R = U_right[1], U_right[2], U_right[3]

    u_L = m_L / max(eps_safe, rho_L)
    u_R = m_R / max(eps_safe, rho_R)

    p_L = (gamma - 1.0) * (E_L - 0.5 * rho_L * u_L^2)
    p_R = (gamma - 1.0) * (E_R - 0.5 * rho_R * u_R^2)

    c_L = sqrt(gamma * max(eps_safe, p_L) / max(eps_safe, rho_L))
    c_R = sqrt(gamma * max(eps_safe, p_R) / max(eps_safe, rho_R))

    return max(abs(u_L) + c_L, abs(u_R) + c_R)
end

function numerical_flux_llf(U_left::AbstractVector, U_right::AbstractVector, gamma::Float64)
    F_left  = physical_flux(U_left,  gamma)
    F_right = physical_flux(U_right, gamma)
    s = max_wave_speed(U_left, U_right, gamma)
    return 0.5 .* (F_left .+ F_right) .- 0.5 .* s .* (U_right .- U_left)
end

# ==============================================================================
# SECTION 2 — Test case definitions (Project, Section 2)
# ==============================================================================

Base.@kwdef struct EulerTestCase
    T::Float64
    L::Float64
    gamma::Function                       # omega -> gamma
    bc::String                            # "periodic" | "neumann" | "custom"
    ic::Function                          # (x, omega, L) -> (rho, u, p)
    bc_left::Union{Nothing,Function} = nothing   # (t, omega) -> (rho, u, p), only for "custom"
end

# 2.2 (i): periodic, random bump location
exercise_2_2_i = EulerTestCase(
    T = 0.5, L = 1.0,
    gamma = omega -> 1.2,
    bc = "periodic",
    ic = (x, omega, L) -> (1.0 + exp(-20.0 * (x - L*omega)^2), 1.0, 10.0)
)

# 2.2 (ii): Neumann, random shock location
exercise_2_2_ii = EulerTestCase(
    T = 0.2, L = 1.0,
    gamma = omega -> 1.4,
    bc = "neumann",
    ic = function (x, omega, L)
        x < omega ? (1.0, 0.0, 1.0) : (0.125, 0.0, 0.1)
    end
)

# 2.3 (i): periodic, random gamma, deterministic IC
exercise_2_3_i = EulerTestCase(
    T = 0.5, L = 1.0,
    gamma = omega -> 1.1 + 0.5*omega,
    bc = "periodic",
    ic = (x, omega, L) -> (1.0 + exp(-20.0 * (x - L/2)^2), 1.0, 10.0)
)

# 2.3 (ii): Neumann, random gamma, deterministic Sod IC
exercise_2_3_ii = EulerTestCase(
    T = 0.2, L = 1.0,
    gamma = omega -> 1.1 + 0.5*omega,
    bc = "neumann",
    ic = function (x, omega, L)
        x < L/2 ? (1.0, 0.0, 1.0) : (0.125, 0.0, 0.1)
    end
)

# 2.4 (i): custom left boundary, both rho_L and p_L random-in-time
exercise_2_4_i = EulerTestCase(
    T = 1.0, L = 1.0,
    gamma = omega -> 1.4,
    bc = "custom",
    ic = function (x, omega, L)
        x < L/2 ? (1.0, 2.0, 1.0) : (0.7, 1.0, 0.7)
    end,
    bc_left = (t, omega) -> (1.0 + exp(-3.0*omega*t), 2.0, 1.0 + exp(-3.0*omega*t))
)

# 2.4 (ii): custom left boundary, only p_L random-in-time
exercise_2_4_ii = EulerTestCase(
    T = 1.0, L = 1.0,
    gamma = omega -> 1.4,
    bc = "custom",
    ic = function (x, omega, L)
        x < L/2 ? (1.0, 2.0, 1.0) : (0.7, 1.0, 0.7)
    end,
    bc_left = (t, omega) -> (1.0, 2.0, 1.0 + exp(-3.0*omega*t))
)

# ==============================================================================
# SECTION 3 — Initial condition & boundary conditions
# ==============================================================================

function setup_initial_condition(n::Int, testcase::EulerTestCase; omega::Float64 = 0.5)
    dx = testcase.L / n
    U = zeros(3, n + 2)
    for i in 2:n+1
        x = (i - 1.5) * dx
        rho, u, p = testcase.ic(x, omega, testcase.L)
        U[:, i] = primitive_to_conservative(rho, u, p, testcase.gamma(omega))
    end
    return U
end

function apply_boundary_conditions!(U::Matrix, testcase::EulerTestCase, t::Float64, omega::Float64)
    if testcase.bc == "periodic"
        U[:, 1]   .= U[:, end-1]
        U[:, end] .= U[:, 2]
    elseif testcase.bc == "neumann"
        U[:, 1]   .= U[:, 2]
        U[:, end] .= U[:, end-1]
    elseif testcase.bc == "custom"
        rho, u, p = testcase.bc_left(t, omega)
        U[:, 1] = primitive_to_conservative(rho, u, p, testcase.gamma(omega))
        U[:, end] .= U[:, end-1]
    else
        error("Unknown boundary condition $(testcase.bc)")
    end
end

# ==============================================================================
# SECTION 4 — Finite volume update (deterministic solve for one omega)
# ==============================================================================

function FV_rhs(U::Matrix, dx::Float64, n::Int, gamma::Float64)
    interface_fluxes = zeros(3, n + 1)
    for i in 1:n+1
        interface_fluxes[:, i] = numerical_flux_llf(U[:, i], U[:, i+1], gamma)
    end
    rhs = zeros(size(U))
    for i in 2:n+1
        rhs[:, i] = -(1.0/dx) .* (interface_fluxes[:, i] .- interface_fluxes[:, i-1])
    end
    return rhs
end

function cfl_timestep(U::Matrix, n::Int, cfl_parameter::Float64, dx::Float64, gamma::Float64)
    s_max = 0.0
    for i in 1:n+1
        s_max = max(s_max, max_wave_speed(U[:, i], U[:, i+1], gamma))
    end
    return s_max == 0.0 ? Inf : cfl_parameter * dx / s_max
end

"""
Solve the deterministic problem for a fixed omega.
Returns final state U_final (3 x (n+2), with ghost cells) and, if save_history=true,
the time history (times, list of real-cell density vectors) for time-dependent plots.
"""
function solve_euler_for_omega(n::Int, testcase::EulerTestCase, omega::Float64;
                                cfl_parameter::Float64 = 0.8, save_history::Bool = false)
    dx = testcase.L / n
    gamma = testcase.gamma(omega)
    t_end = testcase.T

    U = setup_initial_condition(n, testcase; omega = omega)
    t = 0.0

    times = Float64[t]
    rho_history = [copy(U[1, 2:n+1])]

    while t < t_end
        apply_boundary_conditions!(U, testcase, t, omega)
        dt = cfl_timestep(U, n, cfl_parameter, dx, gamma)
        if t + dt > t_end
            dt = t_end - t
        end
        rhs = FV_rhs(U, dx, n, gamma)
        U = U .+ dt .* rhs
        t += dt
        if save_history
            push!(times, t)
            push!(rho_history, copy(U[1, 2:n+1]))
        end
    end
    apply_boundary_conditions!(U, testcase, t, omega)

    return U, times, rho_history
end

# ==============================================================================
# SECTION 5 — Collocation nodes in omega
# ==============================================================================

function uniform_omegas(M::Int)
    domega = 1.0 / M
    return collect(range(domega/2, 1.0 - domega/2, length = M))
end

"Chebyshev-Lobatto nodes on [0,1] -- used as polynomial collocation points (no external package needed)."
function chebyshev_lobatto_omegas(M::Int)
    if M == 1
        return [0.5]
    end
    k = 0:(M-1)
    x = cos.(pi .* k ./ (M-1))            # in [-1,1], endpoints included
    x = sort(x)
    return 0.5 .* (x .+ 1.0)               # map to [0,1]
end

# ==============================================================================
# SECTION 6 — Reconstruction in omega: constant / cubic / polynomial
# ==============================================================================

function constant_reconstruct(omega_eval::Float64, nodes::Vector{Float64}, values::Vector{Float64})
    idx = argmin(abs.(nodes .- omega_eval))
    return values[idx]
end

function cubic_reconstruct(omega_eval::Float64, nodes::Vector{Float64}, values::Vector{Float64})
    n_pts = length(nodes)
    if n_pts < 2
        return values[1]
    end
    k = min(3, n_pts - 1)
    spline = Spline1D(nodes, values; k = k, s = 0.0)
    return spline(omega_eval)
end

"Barycentric Lagrange interpolation -- replaces the Trixi dependency."
function polynomial_reconstruct(omega_eval::Float64, nodes::Vector{Float64}, values::Vector{Float64})
    M = length(nodes)
    if M == 1
        return values[1]
    end
    for j in 1:M
        if isapprox(omega_eval, nodes[j]; atol = 1e-12)
            return values[j]
        end
    end
    w = ones(M)
    for j in 1:M, k in 1:M
        if k != j
            w[j] /= (nodes[j] - nodes[k])
        end
    end
    num = 0.0; den = 0.0
    for j in 1:M
        t = w[j] / (omega_eval - nodes[j])
        num += t * values[j]
        den += t
    end
    return num / den
end

function reconstruct_value(omega_eval::Float64, nodes::Vector{Float64}, values::Vector{Float64}, method::String)
    method == "constant"   && return constant_reconstruct(omega_eval, nodes, values)
    method == "cubic"      && return cubic_reconstruct(omega_eval, nodes, values)
    method == "polynomial" && return polynomial_reconstruct(omega_eval, nodes, values)
    error("Unknown reconstruction method: $method")
end

# ==============================================================================
# SECTION 7 — Stochastic collocation driver
# ==============================================================================

"""
Solve the deterministic problem at M collocation nodes in omega.
Returns: omega_nodes, x_cells, rho_data (Vector of density vectors, one per node, final time)
"""
function stochastic_collocation_driver(n::Int, testcase::EulerTestCase, M::Int; node_type::String = "uniform")
    omega_nodes = node_type == "uniform" ? uniform_omegas(M) : chebyshev_lobatto_omegas(M)
    x_cells = [(i - 0.5) * testcase.L / n for i in 1:n]

    rho_data = Vector{Vector{Float64}}(undef, M)
    for j in 1:M
        U_final, _, _ = solve_euler_for_omega(n, testcase, omega_nodes[j])
        rho_data[j] = copy(U_final[1, 2:n+1])
    end
    return omega_nodes, x_cells, rho_data
end

"Build a (ncells x length(omega_fine)) density surface using the chosen reconstruction method."
function build_density_surface(omega_fine::Vector{Float64}, omega_nodes::Vector{Float64},
                                 rho_data::Vector{Vector{Float64}}, method::String)
    ncells = length(rho_data[1])
    K = length(omega_fine)
    surf = zeros(ncells, K)
    for i in 1:ncells
        y_data = [rho_data[j][i] for j in eachindex(rho_data)]
        for k in 1:K
            surf[i, k] = reconstruct_value(omega_fine[k], omega_nodes, y_data, method)
        end
    end
    return surf
end

# ==============================================================================
# SECTION 8 — Statistics and Lp errors
# ==============================================================================

function compute_statistics(surface::Matrix{Float64})
    return vec(mean(surface, dims = 2)), vec(var(surface, dims = 2))
end

function compute_lp_error(A, B, p)
    D = abs.(A .- B)
    p == 1   && return mean(D)
    p == 2   && return sqrt(mean(D.^2))
    p == Inf && return maximum(D)
    error("Unsupported p = $p")
end

function compute_eoc(Ms::Vector{Int}, errs::Vector{Float64})
    eoc = [NaN]
    for k in 2:length(Ms)
        e = (errs[k] <= 0 || errs[k-1] <= 0) ? NaN :
            log(errs[k-1]/errs[k]) / log(Ms[k]/Ms[k-1])
        push!(eoc, e)
    end
    return eoc
end

# ==============================================================================
# SECTION 9 — Convergence study (reference-based AND Cauchy), L1/L2/Linf
# ==============================================================================

function convergence_study(n::Int, testcase::EulerTestCase, M_values::Vector{Int};
                            node_type::String = "uniform", method::String = "cubic",
                            M_ref::Int = 64,                         # <- new, separate from M_values
                            omega_plot = collect(range(0.0, 1.0, length = 200)))

    om_ref, _, rho_ref_data = stochastic_collocation_driver(n, testcase, M_ref; node_type = node_type)
    rho_ref = build_density_surface(omega_plot, om_ref, rho_ref_data, method)

    L1 = Float64[]; L2 = Float64[]; Linf = Float64[]
    for M in M_values
        om, _, rho_data = stochastic_collocation_driver(n, testcase, M; node_type = node_type)
        surf = build_density_surface(omega_plot, om, rho_data, method)
        push!(L1,   compute_lp_error(surf, rho_ref, 1))
        push!(L2,   compute_lp_error(surf, rho_ref, 2))
        push!(Linf, compute_lp_error(surf, rho_ref, Inf))
    end
    return (M_values = M_values, L1 = L1, L2 = L2, Linf = Linf)
end

"Cauchy-sequence convergence: compare consecutive M_k, M_{k+1} (no fixed reference)."
function cauchy_convergence_study(n::Int, testcase::EulerTestCase, M_values::Vector{Int};
                                   node_type::String = "uniform", method::String = "cubic",
                                   omega_plot = collect(range(0.0, 1.0, length = 200)))
    Ms = sort(M_values)
    surfaces = Dict{Int, Matrix{Float64}}()
    for M in Ms
        om, _, rho_data = stochastic_collocation_driver(n, testcase, M; node_type = node_type)
        surfaces[M] = build_density_surface(omega_plot, om, rho_data, method)
    end

    M_out = Int[]; L1 = Float64[]; L2 = Float64[]; Linf = Float64[]
    for k in 1:length(Ms)-1
        a, b = surfaces[Ms[k]], surfaces[Ms[k+1]]
        push!(M_out, Ms[k+1])
        push!(L1,   compute_lp_error(b, a, 1))
        push!(L2,   compute_lp_error(b, a, 2))
        push!(Linf, compute_lp_error(b, a, Inf))
    end
    return (M_values = M_out, L1 = L1, L2 = L2, Linf = Linf)
end

function print_convergence_table(res; label::String = "")
    println("=== $label ===")
    @printf("%6s %14s %8s %14s %8s %14s %8s\n", "M", "L1", "EOC", "L2", "EOC", "Linf", "EOC")
    eoc1 = compute_eoc(collect(res.M_values), res.L1)
    eoc2 = compute_eoc(collect(res.M_values), res.L2)
    eoci = compute_eoc(collect(res.M_values), res.Linf)
    for k in eachindex(res.M_values)
        s1 = k==1 ? "--" : @sprintf("%.3f", eoc1[k])
        s2 = k==1 ? "--" : @sprintf("%.3f", eoc2[k])
        si = k==1 ? "--" : @sprintf("%.3f", eoci[k])
        @printf("%6d %14.6e %8s %14.6e %8s %14.6e %8s\n",
                res.M_values[k], res.L1[k], s1, res.L2[k], s2, res.Linf[k], si)
    end
    println()
end

# ==============================================================================
# SECTION 10 — MAIN: run experiments and produce plots
# ==============================================================================

function run_experiment_2_2(testcase::EulerTestCase, n::Int, name::String)
    println("\n--- Exercise $name ---")
    omega_fine = collect(range(0.0, 1.0, length = 200))
    M_surface = 12

    om_u, x_cells, rho_u = stochastic_collocation_driver(n, testcase, M_surface; node_type = "uniform")
    om_p, _,       rho_p = stochastic_collocation_driver(n, testcase, M_surface; node_type = "lobatto")

    surf_const = build_density_surface(omega_fine, om_u, rho_u, "constant")
    surf_cubic = build_density_surface(omega_fine, om_u, rho_u, "cubic")
    surf_poly  = build_density_surface(omega_fine, om_p, rho_p, "polynomial")

    # --- mean / variance plot (using cubic reconstruction) ---
    mean_rho, var_rho = compute_statistics(surf_cubic)
    p1 = plot(x_cells, mean_rho, ribbon = sqrt.(var_rho), lw = 2,
              xlabel = "x", ylabel = "rho", title = "$name: mean density +/- std", label = "mean")
    savefig(p1, joinpath(FIGDIR, "$(name)_mean_variance.png"))

    # --- comparison of ansatz spaces at one fixed omega slice ---
    k_mid = length(omega_fine) ÷ 2
    p2 = plot(x_cells, surf_const[:, k_mid], lw=2, label="constant",
              xlabel="x", ylabel="rho", title="$name: reconstruction comparison (omega=$(round(omega_fine[k_mid],digits=2)))")
    plot!(p2, x_cells, surf_cubic[:, k_mid], lw=2, label="cubic")
    plot!(p2, x_cells, surf_poly[:, k_mid], lw=2, label="polynomial")
    savefig(p2, joinpath(FIGDIR, "$(name)_ansatz_comparison.png"))

    # --- density surface heatmap over (x, omega) ---
    p3 = heatmap(omega_fine, x_cells, surf_cubic, xlabel="omega", ylabel="x",
                 title="$name: rho(T,x,omega) [cubic]")
    savefig(p3, joinpath(FIGDIR, "$(name)_surface_heatmap.png"))

    # --- convergence study, all 3 ansatz spaces, L2 norm ---
    M_values = [2, 4, 8, 16]
    conv_const = convergence_study(n, testcase, M_values; node_type="uniform", method="constant")
    conv_cubic = convergence_study(n, testcase, M_values; node_type="uniform", method="cubic")
    conv_poly  = convergence_study(n, testcase, M_values; node_type="lobatto", method="polynomial")

    print_convergence_table(conv_const; label="$name constant (vs reference)")
    print_convergence_table(conv_cubic; label="$name cubic (vs reference)")
    print_convergence_table(conv_poly;  label="$name polynomial (vs reference)")

    cauchy_const = cauchy_convergence_study(n, testcase, M_values; node_type="uniform", method="constant")
    cauchy_cubic = cauchy_convergence_study(n, testcase, M_values; node_type="uniform", method="cubic")
    cauchy_poly  = cauchy_convergence_study(n, testcase, M_values; node_type="lobatto", method="polynomial")

    print_convergence_table(cauchy_const; label="$name constant (Cauchy)")
    print_convergence_table(cauchy_cubic; label="$name cubic (Cauchy)")
    print_convergence_table(cauchy_poly;  label="$name polynomial (Cauchy)")

    p4 = plot(conv_const.M_values, conv_const.L2, marker=:circle, lw=2, xscale=:log10, yscale=:log10,
              label="constant", xlabel="M", ylabel="L2 error (rho)", title="$name: convergence vs reference")
    plot!(p4, conv_cubic.M_values, conv_cubic.L2, marker=:circle, lw=2, label="cubic")
    plot!(p4, conv_poly.M_values,  conv_poly.L2,  marker=:circle, lw=2, label="polynomial")
    savefig(p4, joinpath(FIGDIR, "$(name)_convergence_L2.png"))

    println("Figures saved with prefix '$name' in $FIGDIR/")
end

function run_experiment_2_3(testcase::EulerTestCase, n::Int, name::String)
    println("\n--- Exercise $name (random gamma) ---")
    M_values = [2, 4, 8, 16]

    for method in ("constant", "cubic", "polynomial")
        nt = method == "polynomial" ? "lobatto" : "uniform"
        res_ref    = convergence_study(n, testcase, M_values; node_type=nt, method=method)
        res_cauchy = cauchy_convergence_study(n, testcase, M_values; node_type=nt, method=method)
        print_convergence_table(res_ref;    label="$name $method (vs reference)")
        print_convergence_table(res_cauchy; label="$name $method (Cauchy)")
    end

    # one comparison plot, L2 norm, all methods, reference-based
    p = plot(xscale=:log10, yscale=:log10, xlabel="M", ylabel="L2 error (rho)",
              title="$name: convergence (random gamma)")
    for method in ("constant", "cubic", "polynomial")
        nt = method == "polynomial" ? "lobatto" : "uniform"
        res = convergence_study(n, testcase, M_values; node_type=nt, method=method)
        plot!(p, res.M_values, res.L2, marker=:circle, lw=2, label=method)
    end
    savefig(p, joinpath(FIGDIR, "$(name)_convergence_L2.png"))
    println("Figure saved: $(name)_convergence_L2.png")
end

function run_experiment_2_4(testcase::EulerTestCase, n::Int, name::String)
    println("\n--- Exercise $name (random boundary data) ---")
    omegas_to_show = [0.1, 0.5, 1.0, 2.0]

    p = plot(xlabel="x", ylabel="rho", title="$name: final density for several omega")
    for omega in omegas_to_show
        U_final, times, rho_hist = solve_euler_for_omega(n, testcase, omega; save_history=true)
        x_cells = [(i - 0.5) * testcase.L / n for i in 1:n]
        plot!(p, x_cells, U_final[1, 2:n+1], lw=2, label="omega=$omega")
    end
    savefig(p, joinpath(FIGDIR, "$(name)_density_profiles.png"))
    println("Figure saved: $(name)_density_profiles.png")
end

# ------------------------------------------------------------------------------
# RUN EVERYTHING
# ------------------------------------------------------------------------------

function main()
    n = 200   # spatial cells (increase for production runs, e.g. 400-800)

    run_experiment_2_2(exercise_2_2_i,  n, "ex2_2_i")
    run_experiment_2_2(exercise_2_2_ii, n, "ex2_2_ii")

    run_experiment_2_3(exercise_2_3_i,  n, "ex2_3_i")
    run_experiment_2_3(exercise_2_3_ii, n, "ex2_3_ii")

    run_experiment_2_4(exercise_2_4_i,  n, "ex2_4_i")
    run_experiment_2_4(exercise_2_4_ii, n, "ex2_4_ii")

    println("\nAll done. Figures are in: $(abspath(FIGDIR))")
end

main()