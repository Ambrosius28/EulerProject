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

#! another max_wave_speed function for DGSEM
function max_wave_speed_DG(U_node::AbstractVector, gamma::Float64) # gucke an jedem Knoten nach der größten Geschwindigkeit 
    ρ, m, E = U_node[1], U_node[2], U_node[3]
    u = m / ρ
    p = (gamma - 1) * (E - 0.5 * ρ * u^2)
    c = sqrt(gamma * max(1e-10, p) / max(1e-10, ρ)) # sicherstellen, dass keine Nullwerte kommen
    return abs(u) + c
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
#! change for DGSEM. 
function setup_initial_condition(
    n::Int,
    testcase::EulerTestCase;
    omega::Float64 = 0.5) 

    basis = LobattoLegendreBasis(3) #! use polydeg=3 for the DGSEM
    M = Diagonal(basis.weights)
    D = basis.derivative_matrix

    nodes = length(basis.nodes)
    X = zeros(nodes, n) # n is the number of intervals

    dx = testcase.L / n

    for l in 1:n
        # left boundary of element l 
        x_l = 0.0 + (l - 1) * dx # always start at 0.0

        # Mapping of the basis.nodes (ξ_i) onto the physical nodes X_i_l
        # x(ξ) = x_l + J * (1 + ξ), where J = dx/2 is the half of the cell length (S.75 in Numerics PDE script).
        X[:,l] = x_l .+ (dx/2.0) .* (1.0 .+ basis.nodes)
    end

    #U = zeros(3, n + 2)
    U = zeros(3, nodes, n)

    for l in 1:n
        for i in 1:nodes
            x = X[i,l]
            #x = (i - 1.5) * dx
            rho, u, p = testcase.ic(x, omega, testcase.L)
            U[:,i,l] = primitive_to_conservative(rho, u, p, testcase.gamma(omega))
        end
    end

    return U, X, basis, M, D #! for DGSEM: X, basis, M and D return as well
end

function get_boundary_state(
    U,
    testcase::EulerTestCase,
    t,
    omega,
    side::Symbol)
    # side :left or :right
    if testcase.bc == "periodic"
        return side == :left ? U[:, end, end] : U[:, 1, 1]
    elseif testcase.bc == "neumann"
        return side == :left ? U[:, 1, 1] : U[:, end, end]
    elseif testcase.bc == "custom"
        if side == :left
            rho, u, p = testcase.bc_left(t, omega)
            return primitive_to_conservative(rho, u, p, testcase.gamma(omega))
        else
            return U[:, end, end] # Neumann for right
        end
    end
end

# ==============================================================================
# SECTION 4 — Finite-volume update and deterministic solve for one omega
# ==============================================================================

function get_neighbor_means(mean_array, l, elements, testcase::EulerTestCase)
    # Standard Neighbors
    left_idx = (l == 1) ? (testcase.bc == "periodic" ? elements : l) : l - 1
    right_idx = (l == elements) ? (testcase.bc == "periodic" ? 1 : l) : l + 1
    
    # handle the special boundary conditions
    if l == 1 && testcase.bc == "neumann"
        left_idx = 1 
    elseif l == elements && testcase.bc == "neumann"
        right_idx = elements
    elseif l == 1 && testcase.bc == "custom"
        # Treat it like the Neumann case
        left_idx = 1 
    elseif l == elements && testcase.bc == "custom"
        right_idx = elements
    end
    
    return mean_array[:, left_idx], mean_array[:, right_idx]
end

function minmod(a1::Number,a2::Number,a3::Number)
    if sign(a1) == sign(a2) && sign(a1) == sign(a3)
        return sign(a1) * min(abs(a1), abs(a2), abs(a3))
    else
        return 0.0
    end
end

function minmod_corrected(a1::Number,a2::Number,a3::Number,M,dx)
    if abs(a1) <= M * dx^2
        return a1
    else
        return minmod(a1,a2,a3)
    end
end

function apply_limiter!(U,basis,X,dx,testcase) # U: [n_vars, nodes, elements], X[nodes, elements]
    dimension = size(U)
    elements = dimension[3] # How many elements are there?
    nodes = dimension[2] # How many nodes are there per element?
    sum_weights = sum(basis.weights)
    half_length = dx / 2.0
    tol = 1e-5
    M = 1.3 # for the corrected minmod, choose the value heuristically

    mean_array = zeros(3, elements)

    for l in 1:elements
        mean_array[1,l] = sum(basis.weights .* U[1,:,l]) / sum_weights
        mean_array[2,l] = sum(basis.weights .* U[2,:,l]) / sum_weights
        mean_array[3,l] = sum(basis.weights .* U[3,:,l]) / sum_weights
    end

    for l in 1:elements
        m_left, m_right = get_neighbor_means(mean_array, l, elements, testcase)
    
        ρ_mean_left, m_mean_left, E_mean_left = m_left
        ρ_mean_right, m_mean_right, E_mean_right = m_right

        ρ_mean = mean_array[1,l]
        m_mean = mean_array[2,l]
        E_mean = mean_array[3,l]

        # value of the right boundary in the current element: v_{j+1/2}^{-}
        ρ_right_value = U[1,nodes,l]
        m_right_value = U[2,nodes,l]
        E_right_value = U[3,nodes,l]

        # value of the left boundary in the current element: v_{j-1/2}^{+}
        ρ_left_value = U[1,1,l]
        m_left_value = U[2,1,l]
        E_left_value = U[3,1,l]

        # for each of the three variables u_{j+1/2}^{-} and u_{j-1/2}^{+}
        a1_right = ρ_right_value - ρ_mean
        a1_left = ρ_mean - ρ_left_value
        a2 = ρ_mean - ρ_mean_left
        a3 = ρ_mean_right - ρ_mean
        #U_ρ_right_value = ρ_mean + minmod(a1_right, a2, a3)
        #U_ρ_left_value = ρ_mean - minmod(a1_left, a2, a3)
        U_ρ_right_value = ρ_mean + minmod_corrected(a1_right, a2, a3,M,dx)
        U_ρ_left_value = ρ_mean - minmod_corrected(a1_left, a2, a3,M,dx)


        a1_right = m_right_value - m_mean
        a1_left = m_mean - m_left_value
        a2 = m_mean - m_mean_left
        a3 = m_mean_right - m_mean
        #U_m_right_value = m_mean + minmod(a1_right, a2, a3)
        #U_m_left_value = m_mean - minmod(a1_left, a2, a3)
        U_m_right_value = m_mean + minmod_corrected(a1_right, a2, a3,M,dx)
        U_m_left_value = m_mean - minmod_corrected(a1_left, a2, a3,M,dx)

        a1_right = E_right_value - E_mean
        a1_left = E_mean - E_left_value
        a2 = E_mean - E_mean_left
        a3 = E_mean_right - E_mean
        #U_E_right_value = E_mean + minmod(a1_right, a2, a3)
        #U_E_left_value = E_mean - minmod(a1_left, a2, a3)
        U_E_right_value = E_mean + minmod_corrected(a1_right, a2, a3,M,dx)
        U_E_left_value = E_mean - minmod_corrected(a1_left, a2, a3,M,dx)

        x_center_l = sum(X[:,l]) / nodes # Gauss-Legendre quadrature points are symmetric about the midpoint
        
        # condition (ii), if u_{j+1/2}^{-} = v_{j+1/2}^{-} and u_{j-1/2}^{+} = v_{j-1/2}^{+}
        # for ρ
        if (abs(U_ρ_right_value - ρ_right_value) > tol || abs(U_ρ_left_value - ρ_left_value) > tol)
           local_slope = (U[1,nodes,l] - U[1,1,l]) / (X[nodes,l] - X[1,l])
           slope = minmod(local_slope, (ρ_mean_right - ρ_mean) / half_length, (ρ_mean - ρ_mean_left) / half_length)
           # A new linear polynomial is generated
           U[1,:,l] = ρ_mean .+ (X[:,l] .- x_center_l) .* slope
        end

        # for m
        if (abs(U_m_right_value - m_right_value) > tol || abs(U_m_left_value - m_left_value) > tol)
           local_slope = (U[2,nodes,l] - U[2,1,l]) / (X[nodes,l] - X[1,l])
           slope = minmod(local_slope, (m_mean_right - m_mean) / half_length, (m_mean - m_mean_left) / half_length)
           U[2,:,l] = m_mean .+ (X[:,l] .- x_center_l) .* slope
        end

        # for E
        if (abs(U_E_right_value - E_right_value) > tol || abs(U_E_left_value - E_left_value) > tol)
           local_slope = (U[3,nodes,l] - U[3,1,l]) / (X[nodes,l] - X[1,l])
           slope = minmod(local_slope, (E_mean_right - E_mean) / half_length, (E_mean - E_mean_left) / half_length)
           U[3,:,l] = E_mean .+ (X[:,l] .- x_center_l) .* slope
        end
    end
    return U
end

function DGSEM_time_step!(dU::AbstractArray, U::AbstractArray, p, t)
    dU .= 0.0

    basis, M, D, dx, gamma, omega = p # unpack parameters
    weights = basis.weights
    inv_D_T_M = D' * M # for volume term (see below)

    n_vars, nodes, elements = size(U) # dimensions

    #U_new = zeros(size(U_old))
    interface_fluxes = zeros(n_vars, 2, elements)
    for l in 1:elements
        #! flux on right border of element l (interface l -> l+1)
        U_left = U[:,nodes,l]
        if l == elements
            U_right = get_boundary_state(U, testcase, t, omega, :right)
        else
            U_right = U[:, 1, l+1]
        end
        # flux on the right border of l l 
        interface_fluxes[:,2,l] = numerical_flux_llf(U_left, U_right, gamma)    

        #! flux on left border of element l (Interface l-1 -> l)
        if l == 1
            U_left = get_boundary_state(U, testcase, t, omega, :left)
        else
            U_left = U[:, nodes, l-1]
        end
        U_right = U[:,1, l] # U_right (left side of l)
        interface_fluxes[:,1,l] = numerical_flux_llf(U_left, U_right, gamma)
    end

    # 2. Calculation of the discrete time derivative term dU
    for l in 1:elements
        # Calculate the physical flux f(U) at all nodes of the element, nodal basis
        flux_val = zeros(n_vars, nodes)
        for i in 1:nodes
            flux_val[:, i] = physical_flux(U[:, i, l], gamma)
        end

        for v in 1:n_vars # formula page 78 in the script, dU changed in-place
            # Volume Term: M⁻¹ * Dᵀ * M * f(U)
            # Note: inv(M) is 1/weights
            vol = (1.0 ./ weights) .* (inv_D_T_M * flux_val[v, :])
            
            # Surface Term: M⁻¹ * Rᵀ * B * interface_fluxes
            # Nur am ersten und letzten Knoten
            surf_1 = (1.0 / weights[1]) * interface_fluxes[v, 1, l] # 1/w_1 * interface_fluxes_left
            surf_p = (1.0 / weights[nodes]) * interface_fluxes[v, 2, l] # - 1/w_p * interface_fluxes_right

            # Assemble for each variable v
            # Fill all nodes of element l with the result from the volume term.
            dU[v, :, l] .= (2.0 / dx) .* vol
            # Update for the edge nodes now
            dU[v, 1, l]     += (2.0 / dx) * surf_1
            dU[v, nodes, l] -= (2.0 / dx) * surf_p
        end
    end

    return dU # return dU instead of U_new
end

function solve_euler_for_omega(n::Int, testcase::EulerTestCase, omega::Float64;
                               cfl_parameter::Float64 = 0.8,
                               save_history::Bool = false)

    # --- parameters ---
    dx = testcase.L / n
    gamma = testcase.gamma(omega)
    t_end = testcase.T
    polydeg = 3 #! define polydeg for DGSEM
    nodes = polydeg + 1 #! polydeg + 1 for DGSEM

    # --- initial condition ---
    U, X, basis, M, D = setup_initial_condition(n, testcase; omega = omega)

    t = 0.0

    # --- optional storage ---
    rho_history = Vector{Vector{Float64}}()
    m_history = Vector{Vector{Float64}}()
    E_history = Vector{Vector{Float64}}()
    time_history = Float64[]

    # --- time loop ---
    dU = zeros(size(U)) #! initalize dU for DGSEM
    while t < t_end

        # CFL step
        s_max = 0.0
        for l in 1:n
            for i in 1:nodes
                #s_max = max(s_max, max_wave_speed(U[:, i,l], U[:, i+1,l], gamma))
                s_max = max(s_max, max_wave_speed_DG(U[:, i, l], gamma))
            end
        end

        #dt = cfl_parameter * dx / s_max
        #! another CFL condition for DGSEM
        dt = cfl_parameter * dx / ((2 * polydeg + 1) * s_max)
        if t + dt > t_end
            dt = t_end - t
        end

        if save_history
            push!(rho_history, copy(U[1,:,:]))
            push!(m_history, copy(U[2,:,:]))
            push!(E_history, copy(U[3,:,:]))
            push!(time_history, t)
        end

        parameters = (basis, M, D, dx, gamma, omega) #! define parameter tupel 

        #! SSP-RK3 for DGSEM
        U_n = copy(U)

        #U = finite_volume_time_step(U, dx, dt, n, gamma)
        DGSEM_time_step!(dU, U, parameters, t) #! for DGSEM (name changed from finite_volume_time_step to DGSEM_time_step!)

        U .= U_n + dt .* dU
        apply_limiter!(U, basis, X, dx, testcase) # The limiter must be applied at every intermediate stage.

        # stage 2
        DGSEM_time_step!(dU, U, parameters, t + dt) # passing 't' isn't actually important in our case, as it's an autonomous problem 
        U .= 0.75 * U_n + 0.25 .* (U + dt .* dU)
        apply_limiter!(U, basis, X, dx, testcase)


        # stage 3
        DGSEM_time_step!(dU, U, parameters, t + 0.5 * dt)
        U .= (1/3) .* U_n  + (2/3) .* (U + dt .* dU)
        apply_limiter!(U, basis, X, dx, testcase)

        t += dt
    end

    return U, rho_history, m_history, E_history, time_history, X
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
    #x_cells = [(i - 0.5) * testcase.L / n for i in 1:n] #!! wichtig: muss beim plotten geändert werden!
    
    solutions = Vector{NamedTuple}(undef, M)
    _, _, _, _, _, X = solve_euler_for_omega(n, testcase, omegas[1])

    for j in 1:M

        ω = omegas[j]

        U, rho_history, m_history, E_history, time_history, X = solve_euler_for_omega(n, testcase, ω)

        solutions[j] = (
            omega = ω,

            rho = copy(vec(U[1,:,:])), #! have to change access
            m   = copy(vec(U[2,:,:])),
            E   = copy(vec(U[3,:,:])),

            rho_history = rho_history,
            m_history = m_history,
            E_history = E_history,
            time_history = time_history
        )
    end

    x_cells = vec(X)

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
