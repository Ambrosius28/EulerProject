# 1D Compressible Euler Equations
# Finite Volume / DG(p=0) solver with stochastic collocation in omega
# ==============================================================================

using LinearAlgebra, Printf, Statistics
using Polynomials
using Plots

# ==============================================================================
# SECTION 1 — Euler flux and wave speed
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
# SECTION 2 — Random data definitions
# ==============================================================================

function gamma_value(type_random::String, bc_type::String, omega::Float64)
    if type_random == "gamma"
        return 1.1 + 0.5 * omega
    elseif type_random == "boundary"
        return 1.4
    elseif bc_type == "periodic"
        return 1.2
    elseif bc_type == "neumann"
        return 1.4
    else
        error("Unknown boundary type: $bc_type")
    end
end

function final_time(type_random::String, bc_type::String)
    if type_random == "boundary"
        return 1.0
    elseif bc_type == "periodic"
        return 0.5
    elseif bc_type == "neumann"
        return 0.2
    else
        error("Unknown boundary type: $bc_type")
    end
end

function primitive_initial_data(x::Float64, L::Float64,
                                type_random::String, bc_type::String, omega::Float64)
    if type_random == "boundary"
        if x < L / 2
            return 1.0, 2.0, 1.0
        else
            return 0.7, 1.0, 0.7
        end
    end

    # Use the real omega only when initial data are random.
    # Otherwise omega=0.5 recovers the original deterministic position L/2.
    omega_init = (type_random == "initial") ? omega : 0.5

    if bc_type == "periodic"
        rho = 1.0 + exp(-20.0 * (x - L * omega_init)^2)
        u = 1.0
        p = 10.0
        return rho, u, p
    elseif bc_type == "neumann"
        if x < L * omega_init
            return 1.0, 0.0, 1.0
        else
            return 0.125, 0.0, 0.1
        end
    else
        error("Unknown boundary type: $bc_type")
    end
end

function left_boundary_state(t::Float64, omega::Float64, gamma::Float64,
                             type_boundary_value::String)
    u_L = 2.0
    p_L = 1.0 + exp(-3.0 * omega * t)

    if type_boundary_value == "variiert"
        rho_L = 1.0 + exp(-3.0 * omega * t)
    elseif type_boundary_value == "konstant"
        rho_L = 1.0
    else
        error("Unknown type_boundary_value: $type_boundary_value")
    end

    return primitive_to_conservative(rho_L, u_L, p_L, gamma)
end

# ==============================================================================
# SECTION 3 — Initial condition and boundary conditions
# ==============================================================================

function setup_initial_condition(n::Int, bc_type::String;
                                 type_random::String="none",
                                 omega::Float64=0.5,
                                 type_boundary_value::String="standard")
    L = 1.0
    dx = L / n
    gamma = gamma_value(type_random, bc_type, omega)
    t_end = final_time(type_random, bc_type)

    U = zeros(3, n + 2)

    for i in 2:n+1
        x = (i - 1.5) * dx
        rho, u, p = primitive_initial_data(x, L, type_random, bc_type, omega)
        U[:, i] = primitive_to_conservative(rho, u, p, gamma)
    end

    apply_boundary_conditions!(U, n, bc_type, gamma, 0.0;
                               type_random=type_random,
                               omega=omega,
                               type_boundary_value=type_boundary_value)

    return U, t_end, dx, gamma
end

function apply_boundary_conditions!(U::Matrix, n::Int, bc_type::String,
                                    gamma::Float64, t::Float64;
                                    type_random::String="none",
                                    omega::Float64=0.5,
                                    type_boundary_value::String="standard")
    if type_random == "boundary"
        U[:, 1] = left_boundary_state(t, omega, gamma, type_boundary_value)
        U[:, n+2] = U[:, n+1]
        return
    end

    if bc_type == "periodic"
        U[:, 1]   = U[:, n+1]
        U[:, n+2] = U[:, 2]
    elseif bc_type == "neumann"
        U[:, 1]   = U[:, 2]
        U[:, n+2] = U[:, n+1]
    else
        error("Unknown boundary type: $bc_type")
    end
end

# ==============================================================================
# SECTION 4 — Finite-volume update and deterministic solve for one omega
# ==============================================================================

function finite_volume_time_step(U_old::Matrix, dx::Float64, dt::Float64,
                                 n::Int, gamma::Float64)
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

function solve_euler_for_omega(n::Int, bc_type::String, omega::Float64;
                               type_random::String="none",
                               type_boundary_value::String="standard",
                               cfl_parameter::Float64=0.8,
                               save_history::Bool=false)
    U, t_end, dx, gamma = setup_initial_condition(n, bc_type;
                                                  type_random=type_random,
                                                  omega=omega,
                                                  type_boundary_value=type_boundary_value)
    t = 0.0

    density_history = Vector{Vector{Float64}}()
    time_history = Float64[]

    while t < t_end
        apply_boundary_conditions!(U, n, bc_type, gamma, t;
                                   type_random=type_random,
                                   omega=omega,
                                   type_boundary_value=type_boundary_value)

        s_max = 0.0
        for i in 1:n+1
            s_max = max(s_max, max_wave_speed(U[:, i], U[:, i+1], gamma))
        end

        dt = cfl_parameter * dx / s_max
        if t + dt > t_end
            dt = t_end - t
        end

        if save_history
            push!(density_history, copy(U[1, 2:n+1]))
            push!(time_history, t)
        end

        U = finite_volume_time_step(U, dx, dt, n, gamma)
        t += dt
    end

    apply_boundary_conditions!(U, n, bc_type, gamma, t;
                               type_random=type_random,
                               omega=omega,
                               type_boundary_value=type_boundary_value)

    x_cells = [(i - 0.5) * dx for i in 1:n]
    return U, x_cells, dx, gamma, t_end, density_history, time_history
end
# ============================================================
#SECTION 5 — Collocation nodes in omega
# ==============================================================================

function uniform_omegas(M::Int)
    return collect(range(1.0 / (2M), 1.0 - 1.0 / (2M), length=M))
end