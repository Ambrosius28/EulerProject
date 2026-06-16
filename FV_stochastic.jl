# 1D Compressible Euler Equations
# Finite Volume / DG(p=0) solver with stochastic collocation in omega
# ==============================================================================

using LinearAlgebra, Printf, Statistics
using Polynomials
using Plots
using Trixi
using Dierckx # For cubic spline interpolation

# ==============================================================================
# SECTION 1 — Euler flux and wave speed
# ==============================================================================

function physical_flux(U::AbstractVector, gamma::Float64)
    rho, m, E = U[1], U[2], U[3]
    u = m / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * u^2)
    return [rho * u, m * u + p, (E + p) * u] # F(U) = [ρu, mu + p, (E+p)u]
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

    c_L = sqrt(gamma * max(eps_safe, p_L) / max(eps_safe, rho_L)) # sound speed left = √(γp_L / ρ_L)
    c_R = sqrt(gamma * max(eps_safe, p_R) / max(eps_safe, rho_R)) # sound speed right = √(γp_R / ρ_R)

    return max(abs(u_L) + c_L, abs(u_R) + c_R)
end

function numerical_flux_llf(U_left::AbstractVector, U_right::AbstractVector, gamma::Float64)
    F_left  = physical_flux(U_left,  gamma)
    F_right = physical_flux(U_right, gamma)
    s = max_wave_speed(U_left, U_right, gamma)
    return 0.5 .* (F_left .+ F_right) .- 0.5 .* s .* (U_right .- U_left) #
end

# ==============================================================================
# SECTION 2 — Definition of test cases
# ==============================================================================
Base.@kwdef struct EulerTestCase
    T::Float64
    L::Float64
    gamma::Function
    bc::String 
    ic::Function

    bc_left::Union{Nothing,Function} = nothing
end

exercise_2_2_i = EulerTestCase(
    T = 0.5,
    L = 1.0,
    gamma = omega -> 1.2,
    bc = "periodic",
    ic = (x, omega, L) -> (1 + exp(-20*(x-L*omega)^2), 1.0, 10.0)
)

exercise_2_2_ii = EulerTestCase(
    T = 0.2,
    L = 1.0,
    gamma = omega -> 1.4,
    bc = "neumann",
    ic = function (x, omega, L)
        if x < omega
            return (1.0, 0.0, 1.0)
        else
            return (0.125, 0.0, 0.1)
        end
    end
)

exercise_2_3_i = EulerTestCase(
    T = 0.5,
    L = 1.0,
    gamma = omega -> 1.1 + 0.5*omega,
    bc = "periodic",
    ic = (x, omega, L) -> (1 + exp(-20*(x-L/2)^2), 1.0, 10.0)
)

exercise_2_3_ii = EulerTestCase(
    T = 0.2,
    L = 1.0,
    gamma = omega -> 1.1 + 0.5*omega,
    bc = "neumann",

    ic = function(x, omega, L)
        if x < L/2
            return (1.0, 0.0, 1.0)
        else
            return (0.125, 0.0, 0.1)
        end
    end
)

exercise_2_4_i = EulerTestCase(
    T = 1.0,
    L = 1.0,

    gamma = ω -> 1.4,
    bc = "custom",

    ic = function(x,ω,L)
        if x < L/2
            return (1.0, 2.0, 1.0)
        else
            return (0.7, 1.0, 0.7)
        end
    end,

    bc_left = (t,ω) -> (
        1.0 + exp(-3.0*ω*t),  # rho
        2.0,                  # u
        1.0 + exp(-3.0*ω*t)   # p
    )
)

exercise_2_4_ii = EulerTestCase(
    T = 1.0,
    L = 1.0,

    gamma = ω -> 1.4,
    bc = "custom",

    ic = function(x,ω,L)
        if x < L/2
            return (1.0, 2.0, 1.0)
        else
            return (0.7, 1.0, 0.7)
        end
    end,

    bc_left = (t,ω) -> (
        1.0,                  # rho
        2.0,                  # u
        1.0 + exp(-3.0*ω*t)   # p
    )
)

# ==============================================================================
# SECTION 3 — Initial condition and boundary conditions
# ==============================================================================
function setup_initial_condition(
    n::Int,
    testcase::EulerTestCase;
    omega::Float64 = 0.5) 

    dx = testcase.L / n
    U = zeros(3, n + 2)

    for i in 2:n+1
        x = (i - 1.5) * dx
        rho, u, p = testcase.ic(x, omega, testcase.L)
        U[:,i] = primitive_to_conservative(rho, u, p, testcase.gamma(omega))
    end

    return U
end

function apply_boundary_conditions!(
    U,
    testcase::EulerTestCase,
    t,
    omega)
    if testcase.bc == "periodic"
        U[:,1]   .= U[:,end-1]
        U[:,end] .= U[:,2]
    elseif testcase.bc == "neumann"
        U[:,1]   .= U[:,2]
        U[:,end] .= U[:,end-1]
    elseif testcase.bc == "custom"
        rho, u, p = testcase.bc_left(t, omega)
        U[:,1] = primitive_to_conservative(rho, u, p, testcase.gamma(omega))
        U[:,end] .= U[:,end-1]
    else
        error("Unknown boundary condition $(testcase.bc)")
    end
end

# ==============================================================================
# SECTION 4 — Finite-volume update and deterministic solve for one omega
# ==============================================================================

function finite_volume_time_step(U_old::Matrix, dx::Float64, dt::Float64, n::Int, gamma::Float64)
    U_new = zeros(size(U_old))
    interface_fluxes = zeros(3, n + 1)

    for i in 1:n+1
        interface_fluxes[:, i] = numerical_flux_llf(U_old[:, i], U_old[:, i+1], gamma)
    end

    for i in 2:n+1
        U_new[:, i] = U_old[:, i] .- (dt / dx) .* (interface_fluxes[:, i] .- interface_fluxes[:, i-1])
    end

    return U_new
end

function solve_euler_for_omega(n::Int, testcase::EulerTestCase, omega::Float64;
                               cfl_parameter::Float64 = 0.8,
                               save_history::Bool = false)

    # --- parameters ---
    dx = testcase.L / n
    gamma = testcase.gamma(omega)
    t_end = testcase.T

    # --- initial condition ---
    U = setup_initial_condition(n, testcase; omega = omega)

    t = 0.0

    # --- optional storage ---
    rho_history = Vector{Vector{Float64}}()
    m_history = Vector{Vector{Float64}}()
    E_history = Vector{Vector{Float64}}()
    time_history = Float64[]

    # --- time loop ---
    while t < t_end

        apply_boundary_conditions!(U, testcase, t, omega)

        # CFL step
        s_max = 0.0
        for i in 1:n+1
            s_max = max(s_max, max_wave_speed(U[:, i], U[:, i+1], gamma))
        end

        dt = cfl_parameter * dx / s_max
        if t + dt > t_end
            dt = t_end - t
        end

        if save_history
            push!(rho_history, copy(U[1, 2:n+1]))
            push!(m_history, copy(U[2, 2:n+1]))
            push!(E_history, copy(U[3, 2:n+1]))
            push!(time_history, t)
        end

        U = finite_volume_time_step(U, dx, dt, n, gamma)
        t += dt
    end

    # final boundary correction
    apply_boundary_conditions!(U, testcase, t, omega)
    return U, rho_history, m_history, E_history, time_history
end

# ==============================================================================
#SECTION 5 — Collocation nodes in omega
# ==============================================================================

function uniform_omegas(M::Int)
    domega = 1.0 / M
    return collect(range(domega / 2.0, 1.0 - domega / 2.0, length=M))
end

function legendre_lobatto_basis(M::Int)
    polydeg_stoch = M - 1
    return LobattoLegendreBasis(polydeg_stoch)
end

function legendre_lobatto_omegas(M::Int)
    if M == 1
        return [0.5]
    end

    nodes_reference = legendre_lobatto_basis(M).nodes
    
    nodes_omega = 0.5 .* (nodes_reference .+ 1.0)

    return collect(nodes_omega)
end

# ==============================================================================
# SECTION 6 — Ansatz-space reconstruction 
# ==============================================================================

function constant_maker(omega::Float64,
                        all_omegas::Vector{Float64},
                        y_data::Vector{Float64})

    idx = argmin(abs.(all_omegas .- omega))
    return y_data[idx]
end

function cubic_maker(omega::Float64, 
                     all_omegas::Vector{Float64}, 
                     y_data::Vector{Float64})
    n_points = length(all_omegas)
    if n_points < 2
        error("Cubic reconstruction requires at least 2 collocation points.")
    end

    k = min(3, n_points - 1)  # Dierckx requires k < length(x)
    spline = Spline1D(all_omegas, y_data; k=k, s=0.0)
    return spline(omega)
end

function polynom_maker(omega::Float64,
                       basis_haupt, #this is the Lobatto–Legendre basis object that contains the nodes and weights for polynomial interpolation
                       y_data::Vector{Float64})

    omega_mapped = 2.0 * omega - 1.0

    interpolation_matrix = Trixi.polynomial_interpolation_matrix(basis_haupt.nodes, [omega_mapped])
    return (interpolation_matrix * y_data)[1]
end

function reconstruct_value(omega::Float64,
                           method::String,
                           all_omegas::Vector{Float64},
                           y_data::Vector{Float64};
                           basis_haupt=nothing)

    if method == "constant"
        return constant_maker(omega, all_omegas, y_data)

    elseif method == "cubic"
        return cubic_maker(omega, all_omegas, y_data)

    elseif method == "polynomial"
        if basis_haupt === nothing
            error("Polynomial reconstruction needs basis_haupt.")
        end
        return polynom_maker(omega, basis_haupt, y_data)
    else
        error("Unknown reconstruction method: $method")
    end
end

# ==============================================================================
# SECTION 7 — Stochastic collocation
# ==============================================================================

function stochastic_collocation_driver(
    n::Int,
    testcase::EulerTestCase,
    M::Int;
    collocation_type::String = "uniform")

    # --------------------------------------------------
    # Choose collocation points
    # --------------------------------------------------
    omegas =
        if collocation_type == "uniform"
            uniform_omegas(M)
        elseif collocation_type == "lobatto"
            legendre_lobatto_omegas(M)
        else
            error("Unknown collocation type: $collocation_type")
        end

    # --------------------------------------------------
    # Solve deterministic problems
    # --------------------------------------------------
    x_cells = [(i - 0.5) * testcase.L / n for i in 1:n]
    
    solutions = Vector{NamedTuple}(undef, M)

    for j in 1:M

        ω = omegas[j]

        U, rho_history, m_history, E_history, time_history = solve_euler_for_omega(n, testcase, ω)

        solutions[j] = (
            omega = ω,

            rho = copy(U[1,2:n+1]),
            m   = copy(U[2,2:n+1]),
            E   = copy(U[3,2:n+1]),

            rho_history = rho_history,
            m_history = m_history,
            E_history = E_history,
            time_history = time_history
        )
    end

    return omegas, x_cells, solutions
end

function reconstruct_stochastic(
    omega_eval::Float64,
    solutions,
    reconstruction_method::String;
    basis_haupt = nothing)

    M = length(solutions)

    all_omegas = [sol.omega for sol in solutions]

    n_cells = length(solutions[1].rho)

    rho_interp = zeros(n_cells)
    m_interp   = zeros(n_cells)
    E_interp   = zeros(n_cells)

    for i in 1:n_cells

        rho_data = [sol.rho[i] for sol in solutions]
        m_data   = [sol.m[i]   for sol in solutions]
        E_data   = [sol.E[i]   for sol in solutions]

        rho_interp[i] = reconstruct_value(
            omega_eval,
            reconstruction_method,
            all_omegas,
            rho_data;
            basis_haupt=basis_haupt #check if this is correct
        )

        m_interp[i] = reconstruct_value(
            omega_eval,
            reconstruction_method,
            all_omegas,
            m_data;
            basis_haupt=basis_haupt
        )

        E_interp[i] = reconstruct_value(
            omega_eval,
            reconstruction_method,
            all_omegas,
            E_data;
            basis_haupt=basis_haupt
        )
    end

    return (
        omega = omega_eval,
        rho = rho_interp,
        m   = m_interp,
        E   = E_interp
    )
end

function reconstruct_surfaces(
    omega_fine,
    solutions,
    reconstruction_method;
    basis_haupt=nothing)

    n = length(solutions[1].rho)
    K = length(omega_fine)

    rho_surface = zeros(n, K)
    m_surface   = zeros(n, K)
    E_surface   = zeros(n, K)

    for k in 1:K

        sol_interp = reconstruct_stochastic(
            omega_fine[k],
            solutions,
            reconstruction_method;
            basis_haupt=basis_haupt
        )

        rho_surface[:,k] .= sol_interp.rho
        m_surface[:,k]   .= sol_interp.m
        E_surface[:,k]   .= sol_interp.E
    end

    return rho_surface, m_surface, E_surface
end

function build_density_surface(
    omega_fine::Vector{Float64},
    solutions,
    reconstruction_method::String;
    basis_haupt = nothing)

    n = length(solutions[1].rho)
    K = length(omega_fine)

    rho_surface = zeros(n, K)

    for k in 1:K

        sol_interp = reconstruct_stochastic(
            omega_fine[k],
            solutions,
            reconstruction_method;
            basis_haupt = basis_haupt
        )

        rho_surface[:, k] .= sol_interp.rho
    end

    return rho_surface
end

# ==============================================================================
# SECTION 8 — Statistics
# ==============================================================================

function compute_statistics_surfaces(
    rho_surface,
    m_surface,
    E_surface)

    return (
        mean_rho = vec(mean(rho_surface, dims=2)),
        mean_m   = vec(mean(m_surface, dims=2)),
        mean_E   = vec(mean(E_surface, dims=2)),

        var_rho  = vec(var(rho_surface, dims=2)),
        var_m    = vec(var(m_surface, dims=2)),
        var_E    = vec(var(E_surface, dims=2))
    )
end

# ==============================================================================
# SECTION 9 - Convergence study in Ω using reconstructed stochastic surfaces
# ==============================================================================

function compute_lp_error(A, B, p)

    D = abs.(A .- B)

    if p == 1
        return mean(D)

    elseif p == 2
        return sqrt(mean(D.^2))

    elseif p == Inf
        return maximum(D)

    else
        error("Unsupported p = $p")
    end
end


function compute_surface_errors(
    rho_surface,
    m_surface,
    E_surface,
    rho_ref,
    m_ref,
    E_ref)

    return (
        rho_L1   = compute_lp_error(rho_surface, rho_ref, 1),
        rho_L2   = compute_lp_error(rho_surface, rho_ref, 2),
        rho_Linf = compute_lp_error(rho_surface, rho_ref, Inf),

        m_L1     = compute_lp_error(m_surface, m_ref, 1),
        m_L2     = compute_lp_error(m_surface, m_ref, 2),
        m_Linf   = compute_lp_error(m_surface, m_ref, Inf),

        E_L1     = compute_lp_error(E_surface, E_ref, 1),
        E_L2     = compute_lp_error(E_surface, E_ref, 2),
        E_Linf   = compute_lp_error(E_surface, E_ref, Inf)
    )
end


function stochastic_convergence_study(
    n::Int,
    testcase::EulerTestCase,
    M_values::Vector{Int};
    collocation_type::String = "uniform",
    reconstruction_method::String = "cubic",
    omega_plot = collect(range(0.0, 1.0, length=200)))

    # ------------------------------------------------------------
    # Reference solution (largest M)
    # ------------------------------------------------------------
    M_ref = maximum(M_values)

    _, _, solutions_ref =
        stochastic_collocation_driver(
            n,
            testcase,
            M_ref;
            collocation_type = collocation_type
        )

    basis_ref = legendre_lobatto_basis(M_ref)

    rho_ref, m_ref, E_ref =
        reconstruct_surfaces(
            omega_plot,
            solutions_ref,
            reconstruction_method;
            basis_haupt = basis_ref
        )

    # ------------------------------------------------------------
    # Error storage
    # ------------------------------------------------------------
    rho_L1   = Float64[]
    rho_L2   = Float64[]
    rho_Linf = Float64[]

    m_L1     = Float64[]
    m_L2     = Float64[]
    m_Linf   = Float64[]

    E_L1     = Float64[]
    E_L2     = Float64[]
    E_Linf   = Float64[]

    # ------------------------------------------------------------
    # Loop over M
    # ------------------------------------------------------------
    for M in M_values

        println("Computing stochastic error for M = $M")

        _, _, solutions =
            stochastic_collocation_driver(
                n,
                testcase,
                M;
                collocation_type = collocation_type
            )

        current_basis =
            reconstruction_method == "polynomial" ?
            legendre_lobatto_basis(M) :
            nothing

        rho_surface, m_surface, E_surface =
            reconstruct_surfaces(
                omega_plot,
                solutions,
                reconstruction_method;
                basis_haupt = current_basis
            )

        errors = compute_surface_errors(
            rho_surface,
            m_surface,
            E_surface,
            rho_ref,
            m_ref,
            E_ref
        )

        push!(rho_L1,   errors.rho_L1)
        push!(rho_L2,   errors.rho_L2)
        push!(rho_Linf, errors.rho_Linf)

        push!(m_L1,     errors.m_L1)
        push!(m_L2,     errors.m_L2)
        push!(m_Linf,   errors.m_Linf)

        push!(E_L1,     errors.E_L1)
        push!(E_L2,     errors.E_L2)
        push!(E_Linf,   errors.E_Linf)
    end

    return (
        M_values = M_values,

        rho_L1   = rho_L1,
        rho_L2   = rho_L2,
        rho_Linf = rho_Linf,

        m_L1     = m_L1,
        m_L2     = m_L2,
        m_Linf   = m_Linf,

        E_L1     = E_L1,
        E_L2     = E_L2,
        E_Linf   = E_Linf
    )
end