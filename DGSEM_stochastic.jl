# 1D Compressible Euler Equations
# Finite Volume / DG(p=0) solver with stochastic collocation in omega
# ==============================================================================

using LinearAlgebra, Printf, Statistics
using Polynomials
using Plots
using Trixi
using Dierckx # For cubic spline interpolation
using Base.Threads # for multi-thread computing in the solver
using StaticArrays

# ==============================================================================
# SECTION 1 — Euler flux and wave speed
# ==============================================================================

function physical_flux!(F::AbstractVector, U::AbstractVector, gamma::Float64)
    rho, m, E = U[1], U[2], U[3]
    u = m / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * u^2)

    F[1] = rho*u
    F[2] = m*u + p
    F[3] = (E+p)*u

    return nothing
end

function primitive_to_conservative!(out::AbstractVector, rho, u, p, gamma)
    out[1] = rho
    out[2] = rho * u
    out[3] = p / (gamma - 1.0) + 0.5 * rho * u^2
    return out
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

function max_wave_speed_DG(U_node::AbstractVector, gamma::Float64) # gucke an jedem Knoten nach der größten Geschwindigkeit
    if length(U_node) != 3
        println("FEHLER: U_node hat Länge ", length(U_node), " statt 3!")
        @show U_node
    end 
    ρ, m, E = U_node[1], U_node[2], U_node[3]
    u = m / ρ
    p = (gamma - 1) * (E - 0.5 * ρ * u^2)
    c = sqrt(gamma * max(1e-10, p) / max(1e-10, ρ)) # sicherstellen, dass keine Nullwerte kommen
    return abs(u) + c
end

function numerical_flux_llf!(
    Fnum,
    F_left,
    F_right,
    U_left,
    U_right,
    gamma)

    physical_flux!(F_left, U_left, gamma)
    physical_flux!(F_right, U_right, gamma)

    s = max_wave_speed(U_left, U_right, gamma)

    for i in 1:3
        Fnum[i] = 0.5 * (F_left[i] + F_right[i]) - 0.5 * s * (U_right[i] - U_left[i])
    end

    return nothing
end

# ==============================================================================
# SECTION 2 — Definition of new types: Parameters, EulerTestCase, DeterministicSolution, StochasticSolution
# ==============================================================================
"""
Parameters for the stochastic Euler solver.
"""
Base.@kwdef mutable struct Parameters  #Base.@kwdef allows for default values and keyword arguments in the constructor (you call them by par.n etc.)
    n::Int 
    M::Int = 3
    M_values::Vector{Int} = [3]
    nomega_fine::Vector{Float64}
    ansatz_space::String = "constant"
    cfl_parameter::Float64 = 0.1 # für DG 0.1
    nsnapshots::Int = 4
end

# TODO: define this as a parametric type with GType, BCType, ICType instead of Function.
Base.@kwdef struct EulerTestCase
    T::Float64
    L::Float64
    gamma::Function
    bc::String 
    ic::Function

    bc_left::Union{Nothing,Function} = nothing
end

"""
It contains: times = [t_1,...,T], U=[U(t_1),...,U(T)] (vector of matrices U(x) evaluated at each time)
"""
struct DeterministicSolution
    times::Vector{Float64}
    U::Vector{Array{Float64, 3}}  #! important: {Float64, 3} for DGSEM!
end

"""
It contains: omegas = [ω_1,...,ω_M], solutions = [U(ω_1),...,U(ω_M)] (vector of deterministic solutions evaluated at each omega)
"""
struct StochasticSolution
    omegas::Vector{Float64}
    solutions::Vector{DeterministicSolution}
end

# ==============================================================================
# SECTION 2.5 — Definition of test cases
# ==============================================================================
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

    basis = LobattoLegendreBasis(2) #! use polydeg=2 for the DGSEM
    M = Diagonal(basis.weights)
    D = basis.derivative_matrix

    nodes = length(basis.nodes)
    X = zeros(nodes, n)

    dx = testcase.L / n

    for l in 1:n
        # left boundary of element l 
        x_l = 0.0 + (l - 1) * dx # always start at 0.0

        # Mapping of the basis.nodes (ξ_i) onto the physical nodes X_i_l
        # x(ξ) = x_l + J * (1 + ξ), where J = dx/2 is the half of the cell length (S.75 in Numerics PDE script).
        X[:,l] = x_l .+ (dx/2.0) .* (1.0 .+ basis.nodes)
    end

    U = zeros(3, nodes, n)
    gamma = testcase.gamma(omega)

    for l in 1:n
        for i in 1:nodes
            x = X[i,l]
            #x = (i - 1.5) * dx
            rho, u, p = testcase.ic(x, omega, testcase.L)
            primitive_to_conservative!(view(U, :, i, l), rho, u, p, gamma)
        end
    end

    return U, X, basis, M, D #! for DGSEM: X, basis, M and D return as well
end

"U, testcase (for bc type), t (for time-dependent bc), omega (for omega-dependent bc and gamma)
-> 
U with just ghost cells updated according to the boundary conditions"
function get_boundary_state!(
    out::AbstractVector, # Der Ziel-Vektor
    U::AbstractArray,
    testcase::EulerTestCase,
    t,
    omega,
    side::Symbol)

    if testcase.bc == "periodic"
        if side == :left
            out .= U[:, end, end]
        else
            out .= U[:, 1, 1]
        end
    elseif testcase.bc == "neumann"
        if side == :left
            out .= U[:, 1, 1]
        else
            out .= U[:, end, end]
        end
    elseif testcase.bc == "custom"
        if side == :left
            rho, u, p = testcase.bc_left(t, omega)
            # here as well in-place:
            primitive_to_conservative!(out, rho, u, p, testcase.gamma(omega))
        else
            out .= U[:, end, end]
        end
    end
    return out
end

function get_neighbor_means!(mean_array::AbstractArray, l, elements, testcase::EulerTestCase)
    # Standard Neighbors
    left_idx = (l == 1) ? (testcase.bc == "periodic" ? elements : l) : l - 1
    right_idx = (l == elements) ? (testcase.bc == "periodic" ? 1 : l) : l + 1
    
    # handle the special boundary conditions
    if l == 1 && (testcase.bc == "neumann" || testcase.bc == "custom") # Treat custom like the Neumann case
        left_idx = 1 
    elseif l == elements && (testcase.bc == "neumann" || testcase.bc == "custom")
        right_idx = elements
    end
    
    return view(mean_array, :, left_idx), view(mean_array, :, right_idx)
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
        m_left, m_right = get_neighbor_means!(mean_array, l, elements, testcase)
    
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
# ==============================================================================
# SECTION 4 — Finite-volume update and deterministic solve for one omega
# ==============================================================================
"""
Local Lax-Friedrichs (Rusanov) numerical flux for the Euler equations between two cells.

U_left, U_right, gamma -> F_num (3x1 vector)
"""
function DGSEM_time_step!(testcase::EulerTestCase,inv_D_T_M, dU::AbstractArray, U::AbstractArray, interface_fluxes,
                    u_bc_left, u_bc_right, flux_left, flux_right, flux_buffer, p, t)
    dU .= 0.0

    basis, M, D, dx, gamma, omega = p # unpack parameters
    weights = basis.weights
    n_vars, nodes, elements = size(U) # dimensions

    #U_new = zeros(size(U_old))
    @inbounds for l in 1:elements # inbounds sagt dem Compiler, dass man sich innerhalb der Grenzen befindet, keine 
         # Überprüfung nötig
                #! flux on right border of element l (interface l -> l+1)
                # U_left = U[:,nodes,l] is now taken as a view direct!
                if l == elements
                    get_boundary_state!(u_bc_right, U, testcase, t, omega, :right)
                    U_right_ref = u_bc_right
                else
                    U_right_ref = view(U,:, 1, l+1)
                end
                # flux on the right border of l l 
                numerical_flux_llf!(view(interface_fluxes,:,2,l), flux_left, flux_right,
                                        view(U,:,nodes,l), U_right_ref, gamma)    

                #! flux on left border of element l (Interface l-1 -> l)
                if l == 1
                    get_boundary_state!(u_bc_left, U, testcase, t, omega, :left)
                    U_left_ref = u_bc_left
                else
                    U_left_ref = view(U,:, nodes, l-1)
                end
                #U_right = U[:,1, l] # U_right (left side of l) now taken as a view direct
                numerical_flux_llf!(view(interface_fluxes,:,1,l), flux_left, flux_right, U_left_ref,
                                            view(U,:,1,l), gamma)
              end

    inv_weights = 1.0 ./ weights
    factor = 2.0 / dx
    # 2. Calculation of the discrete time derivative term dU
    @inbounds for l in 1:elements
                # Calculate the physical flux f(U) at all nodes of the element, nodal basis
                for i in 1:nodes
                    physical_flux!(view(flux_buffer,:,i), view(U,:,i,l), gamma)
                end

                for v in 1:n_vars # formula page 78 in the script, dU changed in-place
                    rhs_v = view(dU, v, :, l)
                    # Volume Term: M⁻¹ * Dᵀ * M * f(U)
                    # Note: inv(M) is 1/weights
                    mul!(rhs_v, inv_D_T_M, view(flux_buffer,v,:)) # Matrixmultiplikation mit MatMul: mul!(C,A,B) C = A * B
                    
                    # Surface Term: M⁻¹ * Rᵀ * B * interface_fluxes
                    # Nur am ersten und letzten Knoten
                    # Assemble for each variable v
                    # Fill all nodes of element l with the result from the volume term.
                    @. rhs_v = factor * inv_weights * rhs_v  # Macro, Loop-Fusion
                    # Update for the edge nodes now
                    dU[v, 1, l]     += factor * (1.0 / weights[1]) * interface_fluxes[v,1,l] # 1/w_1 * interface_fluxes_left
                    dU[v, nodes, l] -= factor * (1.0 / weights[nodes]) * interface_fluxes[v,2,l] # - 1/w_p * interface_fluxes_right
                end
            end

    return dU # return dU instead of U_new
end

"""
Right hand side of the finite-volume update 

U (3x(n+2) matrix), dx, n, gamma -> rhs (3x(n+2) matrix)
"""

"""
FIRST STAGE: Solves the deterministic FV+Explicit Euler problem for a given omega 
and returns the DeterministicSolution(omega, times, U_history).

n, testcase, omega; cfl_parameter=0.8 -> DeterministicSolution(times, U_history)
"""
function solve_euler_for_omega(n::Int, testcase::EulerTestCase, omega::Float64;
                               cfl_parameter::Float64 = 0.1, # für DG 0.1 statt 0.8
                               nsnapshots::Int = 100)

    # --- parameters ---
    dx = testcase.L / n
    gamma = testcase.gamma(omega)
    polydeg = 2 #! define polydeg for DGSEM
    nodes = polydeg + 1 #! polydeg + 1 for DGSEM

    # --- initial condition ---
    U, X, basis, M, D = setup_initial_condition(n, testcase; omega = omega)
    t = 0.0

    # --- storage ---
    times = [0.0]
    U_history = [copy(U[:,:,:])]
    snapshot_times = range(0.0, testcase.T, length=nsnapshots)

    n_vars, nodes, elements = size(U) # dimensions
    interface_fluxes = Array{Float64,3}(undef, n_vars, 2, elements)
    u_bc_left  = Vector{Float64}(undef,3)
    u_bc_right = Vector{Float64}(undef,3)
    flux_left  = Vector{Float64}(undef,3)
    flux_right = Vector{Float64}(undef,3)

    flux_buffer = Matrix{Float64}(undef, n_vars, nodes)

    # --- time loop ---
    dU = zeros(size(U)) #! initalize dU for DGSEM
    inv_D_T_M = D' * M # for the volume term 

    for t_snap in snapshot_times[2:end]
        while t < t_snap

            # CFL step
            s_max = 0.0
            for l in 1:n
                for i in 1:nodes
                    #s_max = max(s_max, max_wave_speed(U[:, i,l], U[:, i+1,l], gamma))
                    s_max = max(s_max, max_wave_speed_DG(view(U,:, i, l), gamma))
                end
            end

            #dt = cfl_parameter * dx / s_max
            #! another CFL condition for DGSEM
            dt = cfl_parameter * dx / ((2 * polydeg + 1) * s_max)
            if t + dt > t_snap
                dt = t_snap - t
            end

            parameters = (basis, M, D, dx, gamma, omega) #! define parameter tupel 

            #! SSP-RK3 for DGSEM
            U_n = copy(U)

            #U = finite_volume_time_step(U, dx, dt, n, gamma)
            DGSEM_time_step!(testcase,inv_D_T_M, dU, U, interface_fluxes, u_bc_left, u_bc_right, 
                            flux_left, flux_right, flux_buffer, parameters, t) 
                            #! for DGSEM (name changed from finite_volume_time_step to DGSEM_time_step!)
            U .= U_n + dt .* dU
            apply_limiter!(U, basis, X, dx, testcase) # The limiter must be applied at every intermediate stage.

            # stage 2
            DGSEM_time_step!(testcase,inv_D_T_M, dU, U, interface_fluxes, u_bc_left, u_bc_right, flux_left, flux_right, flux_buffer,
                         parameters, t + dt) # passing 't' isn't actually important in our case, as it's an autonomous problem 
            U .= 0.75 * U_n + 0.25 .* (U + dt .* dU)
            apply_limiter!(U, basis, X, dx, testcase)

            # stage 3
            DGSEM_time_step!(testcase,inv_D_T_M, dU, U, interface_fluxes, u_bc_left, u_bc_right, flux_left, flux_right, flux_buffer,
                         parameters, t + 0.5 * dt)
            U .= (1/3) .* U_n  + (2/3) .* (U + dt .* dU)
            apply_limiter!(U, basis, X, dx, testcase)

            t += dt
        end

        push!(times, t_snap)
        push!(U_history, copy(U[:,:,:]))
    end

    return DeterministicSolution(times, U_history), X, basis
end


# ==============================================================================
#SECTION 5 — Collocation nodes in omega and loop of the deterministic solver for each omega node
# ==============================================================================
function uniform_omegas(M::Int)
    domega = 1.0 / M
    return collect(range(domega / 2.0, 
                         1.0 - domega / 2.0, 
                         length=M))
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
    
    omega_nodes = 0.5 .* (nodes_reference .+ 1.0)

    return collect(omega_nodes)
end

"""
SECOND STAGE: It computes the loop over a given number M of omegas of the deterministic FV+Explicit Euler solution  
and returns the collocation nodes and the corresponding solutions.

n, testcase, M, node_type (uniform/lobatto) -> StochasticSolution(omega_nodes, deterministic_solutions)
"""
function stochastic_collocation_driver(
    n::Int,
    testcase::EulerTestCase,
    M::Int;
    nodes_type::String = "uniform",
    nsnapshots::Int = 100,
    cfl_parameter::Float64 = 0.1)  # 0.1 for DGSEM

    # --------------------------------------------------
    # Choose collocation points
    # --------------------------------------------------
    omega_nodes =
        if nodes_type == "uniform"
            uniform_omegas(M)
        elseif nodes_type == "lobatto"
            legendre_lobatto_omegas(M)
        else
            error("Unknown node type: $nodes_type")
        end

    # --------------------------------------------------
    # Solve deterministic problems
    # --------------------------------------------------
    
    deterministic_solutions = Vector{DeterministicSolution}(undef, M) #create empty vector of DeterministicSolution to store solutions for each omega

    _, X, basis = solve_euler_for_omega(n,testcase,omega_nodes[1];cfl_parameter = cfl_parameter,nsnapshots = nsnapshots)

    Threads.@threads for m in eachindex(omega_nodes) #parallelized computation

     deterministic_solutions[m], _, _ = solve_euler_for_omega(n,
                                            testcase,
                                            omega_nodes[m];
                                            cfl_parameter = cfl_parameter,
                                            nsnapshots = nsnapshots)
    end

    return StochasticSolution(omega_nodes, deterministic_solutions), X, basis
end

"""
(First stage for old implementation) It is the loop over omega_nodes of FV+Euler-Explicit, but with common time stepping for all omegas. 
It returns a StochasticSolution with the same time steps for all omegas.

n, testcase, M, node_type (uniform/lobatto), reconstruction_method (constant/cubic/polynomial) -> StochasticSolution(omega_nodes, solutions)
"""
function cfl_timestep(U, n, cfl_parameter, dx, gamma)
    s_max = 0.0

    for l in 1:n
        for i in 1:3 # 3 because polydeg = 2
            #@show size(U), i, l
            current_view = view(U,:,i,l)
            s_max = max(s_max, max_wave_speed_DG(current_view, gamma))
        end
    end

    if s_max == 0.0
        return Inf
    end
    
    return cfl_parameter * dx / ((2 * 2 + 1) * s_max) # polydeg = 2
end

function stochastic_collocation_driver_common_dt(
    n::Int,
    M::Int,
    testcase::EulerTestCase;
    nodes_type::String = "uniform",
    cfl_parameter::Float64 = 0.1)

    # --------------------------------------------------
    # Choose collocation points
    # --------------------------------------------------
    omega_nodes =
        if nodes_type == "uniform"
            uniform_omegas(M)
        elseif nodes_type == "lobatto"
            legendre_lobatto_omegas(M)
        else
            error("Unknown node type: $nodes_type")
        end

    # --------------------------------------------------
    # Spatial parameters
    # --------------------------------------------------
    dx    = testcase.L / n
    t_end = testcase.T

    # --------------------------------------------------
    # Initialize StochasticSolution
    # --------------------------------------------------
    solutions = Vector{DeterministicSolution}(undef, M)
    U_work = [zeros(3, 3, n) for _ in 1:M] # second 3 for the number of nodes per cell (hard coded because of polydeg=2)

    _, X, basis, M_matrix, D = setup_initial_condition(n, testcase; omega=omega_nodes[1])
    inv_D_T_M = D' * M_matrix # for the volume term
    dU_all = [zeros(size(U_work[1])) for _ in 1:M] #! initalize dU for DGSEM
    @show size(dU_all[1])

    n_vars, nodes, elements = size(U_work[1]) # dimensions
    interface_fluxes_all = [Array{Float64,3}(undef, n_vars, 2, elements) for _ in 1:M]
    u_bc_left_all  = [Vector{Float64}(undef,3) for _ in 1:M]
    u_bc_right_all = [Vector{Float64}(undef,3) for _ in 1:M]
    flux_left_all  = [Vector{Float64}(undef,3) for _ in 1:M]
    flux_right_all = [Vector{Float64}(undef,3) for _ in 1:M]

    flux_buffer_all = [Matrix{Float64}(undef, n_vars, nodes) for _ in 1:M]

    # Initialize: fill interior from IC, store only interior in history
    for (m, ω) in enumerate(omega_nodes)
        U0, X, basis, M, D = setup_initial_condition(n, testcase; omega=ω)
        U_work[m] .= U0
        solutions[m] = DeterministicSolution([0.0], [copy(U0[:,:,:])]) 
    end

    stochastic = StochasticSolution(omega_nodes, solutions)

    # --------------------------------------------------
    # Global time-stepping loop
    # --------------------------------------------------
    t = 0.0

    while t < t_end

        # Re-pad and apply BCs
        for (m, ω) in enumerate(omega_nodes)
            U_work[m][:,:,:] .= stochastic.solutions[m].U[end] # the time step we're in at the moment
        end

        # Compute adaptive common dt as the minimum (over omega) of the cfl timesteps
        dt_min = Inf
        for (m, ω) in enumerate(omega_nodes)
            γ = testcase.gamma(ω)
            dt_ω = cfl_timestep(U_work[m], n, cfl_parameter, dx, γ)
            dt_min = min(dt_min, dt_ω)
        end

        if dt_min == Inf
            break
        end

        # Ensure final dt reaches exactly t_end
        if t + dt_min > t_end
            dt_min = t_end - t
        end

        # compute the timestep with FV + Explicit Euler
        Threads.@threads for m in 1:length(omega_nodes)
                    ω = omega_nodes[m]
                    γ   = testcase.gamma(ω)
                    parameters = (basis, M, D, dx, γ, ω) #! define parameter tupel
                    U = U_work[m]
                    dU = dU_all[m]
                    U_n = copy(U)

                    #U = finite_volume_time_step(U, dx, dt, n, gamma)
                    DGSEM_time_step!(testcase,inv_D_T_M, dU, U, interface_fluxes_all[m], u_bc_left_all[m], u_bc_right_all[m], 
                                    flux_left_all[m], flux_right_all[m], flux_buffer_all[m], parameters, t) 
                                    #! for DGSEM (name changed from finite_volume_time_step to DGSEM_time_step!)
                    U .= U_n + dt_min .* dU
                    apply_limiter!(U, basis, X, dx, testcase) # The limiter must be applied at every intermediate stage.

                    # stage 2
                    DGSEM_time_step!(testcase,inv_D_T_M, dU, U, interface_fluxes_all[m], u_bc_left_all[m], u_bc_right_all[m], flux_left_all[m],
                                flux_right_all[m], flux_buffer_all[m], parameters, t + dt_min) # passing 't' isn't actually important in our case, as it's an autonomous problem 
                    U .= 0.75 * U_n + 0.25 .* (U + dt_min .* dU)
                    apply_limiter!(U, basis, X, dx, testcase)

                    # stage 3
                    DGSEM_time_step!(testcase,inv_D_T_M, dU, U, interface_fluxes_all[m], u_bc_left_all[m], u_bc_right_all[m], flux_left_all[m],
                                flux_right_all[m], flux_buffer_all[m], parameters, t + 0.5 * dt_min)
                    U .= (1/3) .* U_n  + (2/3) .* (U + dt_min .* dU)
                    apply_limiter!(U, basis, X, dx, testcase)

                    push!(stochastic.solutions[m].U, copy(U[:,:,:]))
                end

        t += dt_min
        for sol in stochastic.solutions
            push!(sol.times, t)
        end
    end

    return stochastic, X
end


# ==============================================================================
# SECTION 6 — Ansatz-space reconstruction 
# ==============================================================================
"omega_eval, omega_nodes, y_data ->  value of y at omega_eval using piecewise constant interpolation"
function constant_maker(omega_eval::Float64,
                        omega_nodes::Vector{Float64},
                        y_data::Vector{Float64})

    idx = argmin(abs.(omega_nodes .- omega_eval))
    return y_data[idx]
end

"omega_eval, omega_nodes, y_data -> value of y at omega_eval using cubic spline interpolation"
function cubic_maker(omega_eval::Float64, 
                     omega_nodes::Vector{Float64}, 
                     y_data::Vector{Float64})
    n_points = length(omega_nodes)
    if n_points < 2
        error("Cubic reconstruction requires at least 2 collocation points.")
    end

    k = min(3, n_points - 1)  # Dierckx requires k < length(x)
    spline = Spline1D(omega_nodes, y_data; k=k, s=0.0)
    return spline(omega_eval)
end

"omega_eval, omega_nodes, y_data -> value of y at omega_eval using polynomial interpolation"
function polynom_maker(omega_eval::Float64,
                       omega_nodes, 
                       y_data::Vector{Float64})

    basis = legendre_lobatto_basis(length(omega_nodes))

    omega_mapped = 2.0 * omega_eval - 1.0

    interpolation_matrix = Trixi.polynomial_interpolation_matrix(basis.nodes, [omega_mapped])
    return (interpolation_matrix * y_data)[1]
end

"""
It just choose to apply one of the three reconstruction methods

omega_eval, omega_nodes, y_data, method (constant/cubic/polynomial) -> value of y at omega_eval using the specified method 
"""
function reconstruct_value(omega_eval::Float64,
                           omega_nodes::Vector{Float64},
                           y_data::Vector{Float64},
                           method::String,)

    if method == "constant"
        return constant_maker(omega_eval, omega_nodes, y_data)

    elseif method == "cubic"
        return cubic_maker(omega_eval, omega_nodes, y_data)

    elseif method == "polynomial"
        return polynom_maker(omega_eval, omega_nodes, y_data)
    else
        error("Unknown reconstruction method: $method")
    end
end

# ==============================================================================
# SECTION 7 — Stochastic collocation
# ==============================================================================
function get_idx_and_alpha(t_target::Number, times::AbstractArray) #! use linear interpolation 
    idx = searchsortedlast(times, t_target) #search last index in times, whose value is smaller or equal than times

    idx = clamp(idx, 1, length(times)-1)

    t0, t1 = times[idx], times[idx+1] # time points from the array, which enclose the t_target

    dt = t1 - t0 
    α = dt > 0 ? (t_target - t0) / dt : 0.0 # how much is t_target between t0 and t1

    return idx, α
end

"omega_eval, stochastic, reconstruction_method -> DeterministicSolution at omega_eval using the specified reconstruction method"
function evaluate_at_omega(
    omega_eval::Float64,
    stochastic::StochasticSolution,
    reconstruction_method::String,
    M::Int)

    omega_nodes = stochastic.omegas
    solutions   = stochastic.solutions
    
    N = length(stochastic.solutions)
    time_control = Int[]
    for i in 1:N
        #println("Time points before interpolation: ", solutions[i].times)
        nu_times = length(solutions[i].times) 
        push!(time_control, nu_times)
    end

    if length(unique(time_control)) > 1
        @warn "Inkonsistente Anzahl an Zeitschritten gefunden!"
    end

    target_times = solutions[max(1, M ÷ 2)].times # times of the M/2 omega solution
    number_timesteps = length(solutions[1].times)
    U_interp = Vector{Array{Float64, 3}}(undef, number_timesteps)
    ncomp, nnodes, nelements = size(solutions[1].U[1])
    data_at_t = zeros(length(solutions))
    #Uj = zeros(ncomp, nnodes, nelements)

    for j in 1:number_timesteps                     #loop over time steps
        t_target = target_times[j] # whole time scale based on the omega solution, where it's taken from. 
        Uj = zeros(ncomp, nnodes, nelements)
        interp_params = [get_idx_and_alpha(t_target, sol.times) for sol in solutions] # returns tupel per omega

        for k in 1:ncomp                            #loop over components (rho, m, E)
            for i in 1:nelements                       #loop over elements
                for l in 1:nnodes                   # loop over nodes
                    #data = [sol.U[j][k,l,i] for sol in solutions] #vector (one component for each omega) of values of U at fixed time j, component k, cell i 
                    for (m, sol) in enumerate(solutions)
                        idx, α = interp_params[m]
                        u0 = sol.U[idx][k,l,i]
                        u1 = sol.U[idx+1][k,l,i]

                        data_at_t[m] = (1.0 - α) * u0 + α * u1
                    end
                    
                    Uj[k,l,i] = reconstruct_value(
                        omega_eval,
                        omega_nodes,
                        data_at_t,
                        reconstruction_method)
                # Uj[k,i] = data[end]    
                end
            end
        end
        U_interp[j] = Uj
    end

    return DeterministicSolution(target_times, U_interp)
end

"""
THIRD STAGE: it returns a StochasticSolution with the reconstructed solutions at the fine omega grid using the specified reconstruction method.

omega_fine, stochastic, reconstruction_method -> StochasticSolution(omega_fine, reconstructed_solutions)
"""
function reconstruct_stochastic_solution(
    omega_fine::Vector{Float64},
    stochastic::StochasticSolution,
    reconstruction_method::String,
    M::Int)

    fine_solutions = DeterministicSolution[]

    for omega in omega_fine

        sol = evaluate_at_omega(omega, stochastic, reconstruction_method,M)
        #println("Time points after interpolation: ", sol.times)
        push!(fine_solutions, sol)

    end

    return StochasticSolution(omega_fine, fine_solutions)
end

"""
stoch::StochasticSolution -> U as a 5-index matrix, where the indices are (component, t, nodes, elements, omega)
"""
function tensorize(stoch::StochasticSolution)

    nω = length(stoch.solutions) #number of omegas_fine
    nt = length(stoch.solutions[1].times) #number of time steps

    nc, nn, ne = size(stoch.solutions[1].U[1]) #number of components, nodes and elements

    U = zeros(nc, nt, nn, ne, nω)

    for k in 1:nω
        for j in 1:nt
            U[:,j,:,:,k] .= stoch.solutions[k].U[j]
        end
    end

    return U
end

function main_old( 
    n::Int,
    M::Int,
    testcase::EulerTestCase,
    omega_fine::Vector{Float64},
    reconstruction_method::String;
    cfl_parameter::Float64 = 0.1)
    
    if reconstruction_method == "constant"
        solution_at_nodes, X = stochastic_collocation_driver_common_dt(n, M, testcase; 
                                                                    nodes_type = "uniform",
                                                                    cfl_parameter = cfl_parameter)
    elseif reconstruction_method == "cubic"
        solution_at_nodes, X = stochastic_collocation_driver_common_dt(n, M, testcase; 
                                                                    nodes_type = "uniform",
                                                                    cfl_parameter = cfl_parameter)
    elseif reconstruction_method == "polynomial"
        solution_at_nodes, X = stochastic_collocation_driver_common_dt(n, M, testcase; 
                                                                    nodes_type = "lobatto",
                                                                    cfl_parameter = cfl_parameter)
    else
        error("Unknown reconstruction method: $reconstruction_method")
    end

    solution = reconstruct_stochastic_solution(omega_fine,
                                               solution_at_nodes,
                                               reconstruction_method,M)
    return solution, X
end

"""
MAIN SIMULATION FUNCTION

n, M, testcase, omega_fine, reconstruction_method (constant/cubic/polynomial); cfl_parameter -> StochasticSolution(omega_fine, reconstructed_solutions)
"""
function main(testcase::EulerTestCase, par::Parameters)

    n = par.n
    M = par.M
    omega_fine = par.nomega_fine
    ansatz_space = par.ansatz_space
    cfl_parameter = par.cfl_parameter
    nsnapshots = par.nsnapshots
    println("DEBUG: main() gestartet mit M = ", M)

    nodes_type =
        if ansatz_space == "constant" || ansatz_space == "cubic"
            "uniform"
        elseif ansatz_space == "polynomial"
            "lobatto"
        else
            error("Unknown reconstruction method: $ansatz_space")
        end
   
    #=solution_at_nodes, X = stochastic_collocation_driver_common_dt(
        n,
        M,
        testcase;
        nodes_type = nodes_type,
        cfl_parameter = cfl_parameter,
    )=#

    solution_at_nodes, X, basis = stochastic_collocation_driver(
        n,
        testcase,
        M;
        nodes_type = nodes_type,
        nsnapshots = nsnapshots,
        cfl_parameter = cfl_parameter,
    )
    
    solution = reconstruct_stochastic_solution( # type: StochasticSolution
        omega_fine,
        solution_at_nodes,
        ansatz_space,
        M
    )

    return solution, X, basis
end
