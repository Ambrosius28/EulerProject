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

# ==============================================================================
# SECTION 2 — Definition of test cases
# ==============================================================================
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
    U::Vector{Matrix{Float64}}
end

"""
It contains: omegas = [ω_1,...,ω_M], solutions = [U(ω_1),...,U(ω_M)] (vector of deterministic solutions evaluated at each omega)
"""
struct StochasticSolution
    omegas::Vector{Float64}
    solutions::Vector{DeterministicSolution}
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
"n, testcase, omega -> U (3x(n+2) matrix containing initial data for rho, m, E in the interior cells and ghost cells)"
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

"U, testcase (for bc type), t (for time-dependent bc), omega (for omega-dependent bc and gamma)
-> 
U with just ghost cells updated according to the boundary conditions"
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
"""
Local Lax-Friedrichs (Rusanov) numerical flux for the Euler equations between two cells.

U_left, U_right, gamma -> F_num (3x1 vector)
"""
function numerical_flux_llf(U_left::AbstractVector, U_right::AbstractVector, gamma::Float64)
    F_left  = physical_flux(U_left,  gamma)
    F_right = physical_flux(U_right, gamma)
    s = max_wave_speed(U_left, U_right, gamma)
    return 0.5 .* (F_left .+ F_right) .- 0.5 .* s .* (U_right .- U_left) #
end

"""
Right hand side of the finite-volume update 

U (3x(n+2) matrix), dx, n, gamma -> rhs (3x(n+2) matrix)
"""
function FV_rhs(U::Matrix, dx::Float64, n::Int, gamma::Float64)
    interface_fluxes = zeros(3, n + 1)

    for i in 1:n+1
        interface_fluxes[:, i] =
            numerical_flux_llf(U[:, i], U[:, i+1], gamma)
    end

    rhs = zeros(size(U))

    for i in 2:n+1
        rhs[:, i] = -(1.0 / dx) .* (
            interface_fluxes[:, i] .- interface_fluxes[:, i-1]
        )
    end

    return rhs
end

"U_old, rhs, dt -> U_new after one explicit Euler time step"
function explicit_euler_time_step(U_old::Matrix, rhs::Matrix, dt::Float64)
    U_new = U_old .+ dt .* rhs
    return U_new
end

"U (for computing max wave speed), cfl_parameter, dx, gamma (for computing max wave speed) -> dt"
function cfl_timestep(U, n, cfl_parameter, dx, gamma)
    s_max = 0.0
    
    for i in 1:n+1
        s_max = max(s_max, max_wave_speed(U[:, i], U[:, i+1], gamma)) 
    end

    if s_max == 0.0
        return Inf
    end

    return cfl_parameter * dx / s_max
end

"""
(FIRST STAGE): Solves the deterministic FV+Explicit Euler problem for a given omega 
and returns the DeterministicSolution(omega, times, U_history).

n, testcase, omega; cfl_parameter=0.8 -> DeterministicSolution(omega, times, U_history)
"""
function solver_FV(
    n::Int,
    testcase::EulerTestCase,
    omega::Float64;
    cfl_parameter::Float64 = 0.8)

    # --- parameters ---
    dx = testcase.L / n
    gamma = testcase.gamma(omega)
    t_end = testcase.T

    # --- initial condition ---
    U = setup_initial_condition(n, testcase; omega = omega)
    t = 0.0

    # --- storage ---
    times = Float64[]
    U_history = Matrix{Float64}[]

    
    push!(times, t)
    push!(U_history, copy(U))
  

    # --- time loop ---
    while t < t_end

        apply_boundary_conditions!(U, testcase, t, omega)

        dt = cfl_timestep(U, n, cfl_parameter, dx, gamma)

        if t + dt > t_end
            dt = t_end - t
        end

        rhs = FV_rhs(U, dx, n, gamma)

        U = explicit_euler_time_step(U, rhs, dt)
        t += dt

        push!(times, t)
        push!(U_history, copy(U)) #TODO: maybe push!(U_history, copy(U[:, 2:n+1])) to save memory since ghost cells are not needed for reconstruction?
    end

    apply_boundary_conditions!(U, testcase, t, omega) #TODO: maybe not needed since we won't use ghost cells for reconstruction?
    U_history[end] = copy(U)

    return DeterministicSolution(times, U_history)
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
(SECOND STAGE): It computes the loop over a given number M of omegas of the deterministic FV+Explicit Euler solution  
and returns the collocation nodes and the corresponding solutions.

n, testcase, M, node_type (uniform/lobatto) -> StochasticSolution(omega_nodes, deterministic_solutions)
"""
function stochastic_collocation_driver(
    n::Int,
    testcase::EulerTestCase,
    M::Int;
    nodes_type::String = "uniform")

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
    
      deterministic_solutions = DeterministicSolution[] #create empty vector of DeterministicSolution to store solutions for each omega

    for omega in omega_nodes

        sol = solver_FV(n, testcase, omega)

        push!(deterministic_solutions, sol)
    end

    return StochasticSolution(omega_nodes, deterministic_solutions)
end

"""
FIRST STAGE It is the loop over omega_nodes of FV+Euler-Explicit, but with common time stepping for all omegas. 
It returns a StochasticSolution with the same time steps for all omegas.

n, testcase, M, node_type (uniform/lobatto), reconstruction_method (constant/cubic/polynomial) -> StochasticSolution(omega_nodes, solutions)
"""
function stochastic_collocation_driver_common_dt(
    n::Int,
    M::Int,
    testcase::EulerTestCase;
    nodes_type::String = "uniform",
    cfl_parameter::Float64 = 0.8)

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
    U_work = [zeros(3, n+2) for _ in 1:M]

    # Initialize: fill interior from IC, store only interior in history
    for (m, ω) in enumerate(omega_nodes)
        U0 = setup_initial_condition(n, testcase; omega=ω)
        U_work[m] .= U0
        solutions[m] = DeterministicSolution([0.0], [copy(U0[:, 2:end-1])]) #doesn't store the ghost cells
    end

    stochastic = StochasticSolution(omega_nodes, solutions)

    # --------------------------------------------------
    # Global time-stepping loop
    # --------------------------------------------------
    t = 0.0

    while t < t_end

        # Re-pad and apply BCs
        for (m, ω) in enumerate(omega_nodes)
            U_work[m][:, 2:end-1] .= stochastic.solutions[m].U[end]
            apply_boundary_conditions!(U_work[m], testcase, t, ω)
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
        for (m, ω) in enumerate(omega_nodes)
            γ   = testcase.gamma(ω)
            rhs = FV_rhs(U_work[m], dx, n, γ)
            U_work[m] = explicit_euler_time_step(U_work[m], rhs, dt_min)
            push!(stochastic.solutions[m].U, copy(U_work[m][:, 2:end-1]))
        end

        t += dt_min
        for sol in stochastic.solutions
            push!(sol.times, t)
        end
    end

    return stochastic
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
"omega_eval, stochastic, reconstruction_method -> DeterministicSolution at omega_eval using the specified reconstruction method"
function evaluate_at_omega(
    omega_eval::Float64,
    stochastic::StochasticSolution,
    reconstruction_method::String)

    omega_nodes = stochastic.omegas
    solutions   = stochastic.solutions

    number_timesteps = length(solutions[1].U) #TODO: here we are not considering the adaptive time stepping, for which we have different times for each omega. 

    U_interp = Matrix{Float64}[]
    ncomp, ncells = size(solutions[1].U[1]) 
    Uj = zeros(ncomp, ncells)

    for j in 1:number_timesteps                     #loop over time steps
        for k in 1:ncomp                            #loop over components (rho, m, E)
            for i in 1:ncells                       #loop over cells
                data = [sol.U[j][k,i] for sol in solutions] #vector (one component for each omega) of values of U at fixed time j, component k, cell i 
                Uj[k,i] = reconstruct_value(
                    omega_eval,
                    omega_nodes,
                    data,
                    reconstruction_method)
            end
        end

        push!(U_interp, copy(Uj))
    end

    return DeterministicSolution(solutions[1].times, U_interp)
end

"""
SECOND STAGE (THIRD STAGE): it returns a StochasticSolution with the reconstructed solutions at the fine omega grid using the specified reconstruction method.

omega_fine, stochastic, reconstruction_method -> StochasticSolution(omega_fine, reconstructed_solutions)
"""
function reconstruct_stochastic_solution(
    omega_fine::Vector{Float64},
    stochastic::StochasticSolution,
    reconstruction_method::String)

    fine_solutions = DeterministicSolution[]

    for omega in omega_fine

        sol = evaluate_at_omega(omega, stochastic, reconstruction_method)
        push!(fine_solutions, sol)

    end

    return StochasticSolution(omega_fine, fine_solutions)
end

"""
stoch::StochasticSolution -> U as a 4-index matrix, where the indices are (component, t, x, omega)
"""
function tensorize(stoch::StochasticSolution)

    nω = length(stoch.solutions) #number of omegas_fine
    nt = length(stoch.solutions[1].times) #number of time steps

    nc, nx = size(stoch.solutions[1].U[1]) #number of components and of cells

    U = zeros(nc, nt, nx, nω)

    for k in 1:nω
        for j in 1:nt
            U[:,j,:,k] .= stoch.solutions[k].U[j]
        end
    end

    return U
end

"""
MAIN SIMULATION FUNCTION

n, M, testcase, omega_fine, reconstruction_method (constant/cubic/polynomial); cfl_parameter -> StochasticSolution(omega_fine, reconstructed_solutions)
"""
function main( 
    n::Int,
    M::Int,
    testcase::EulerTestCase,
    omega_fine::Vector{Float64},
    reconstruction_method::String;
    cfl_parameter::Float64 = 0.8)
    
    if reconstruction_method == "constant"
        solution_at_nodes = stochastic_collocation_driver_common_dt(n, M, testcase; 
                                                                    nodes_type = "uniform",
                                                                    cfl_parameter = cfl_parameter)
    elseif reconstruction_method == "cubic"
        solution_at_nodes = stochastic_collocation_driver_common_dt(n, M, testcase; 
                                                                    nodes_type = "uniform",
                                                                    cfl_parameter = cfl_parameter)
    elseif reconstruction_method == "polynomial"
        solution_at_nodes = stochastic_collocation_driver_common_dt(n, M, testcase; 
                                                                    nodes_type = "lobatto",
                                                                    cfl_parameter = cfl_parameter)
    else
        error("Unknown reconstruction method: $reconstruction_method")
    end

    solution = reconstruct_stochastic_solution(omega_fine,
                                               solution_at_nodes,
                                               reconstruction_method)
    return solution
end