# one-dimensional, compressible Euler equations
# ∂tσ + ∂xm = 0, describes mass conservation (m = ρ*u), ρ mass density, u velocity, m momentum density
# mass cannot simply disappear or appear from nothing
# ∂tm + ∂x (um) + ∂xp = 0, describes momentum conservation
# ∂tE + ∂x((E + p)u) = 0, describes energy conservation
# p(t,x) = (γ−1)(E(t,x) −ρ(t,x) * (u(t,x)^2/2)), γ adiabatic coefficient: how much pressure rises when compressing the
# gas without heat escaping
# E = p / (γ − 1) + 1/2 * ρ*u^2   #! S.295 RANDALL J. LEVEQUE Finite Volume Methods

# no viscosity (no internal friction, viscosity)
# compressible system, density ρ can change
# hyperbolic system: waveform, the wave remains preserved and moves onward, sharp structures remain preserved
# ∂t + ∂x = 0 typical for hyperbolic systems

# solution vector is (ρ, m, E) #! This is the state Q

###########################################################################################################################
# shocks are problems in hyperbolic conservation laws. once a shock wave appears, the function at the
# discontinuity is no longer differentiable. the derivative would be "infinite" there
# there are classical solutions (C^1 differentiable) and weak solutions (multiply by a test function + integrate)
# there are multiple mathematically correct weak solutions, but these can be physically meaningless.
# entropy conditions filter out the unphysical solutions
# the notion of solution is not well-posed. 

#! finite volume two assumptions: the shock is therefore treated as a discontinuity that sits exactly at the boundary 
#! between two cells
#! in finite volume it is assumed that the solution is constant inside a cell. all information 
#! of the PDE is distributed over an average in one cell.

################################################################################################
#! limiter:
# where the solution is smooth, the limiter is not used; where jumps and shocks occur, the limiter irons the 
# slope/curvature of the polynomials smoothly toward piecewise constant.
# the limiter ensures that the sign condition in Harten's lemma is satisfied.

# blueprint:
# 2.5. The Non-Linear Stability of the RKDG Method, p.196, Cockburn and Shu
# alpha and beta parameters for RK: Table 2.1 p.190
# limiter: corrected minmod function, pp.193-195
# confirmation for our CFL number: p.190
# there is a CFL for L2 stability and TV (Total Variation) stability. since it is much weaker for TV,
# you can simply consult Table 2.2 on p.191 for the CFL number

#################################################################################################
#! stochastic variable part
# different variables are now random variables, they depend on the value ω, which can take values between 0 and 1.
# an outer loop is placed around the code to iterate through the collocation points.
# the individual solutions are then welded together into one solution using piecewise constant functions, cubic splines and polynomials.
# for splines, it is best to use equally spaced values of ω; for polynomials use GLL nodes.
# once the initial values depend on ω, then the adiabatic coefficient γ
# for the adiabatic coefficient: every value of ω between 0 and 1 is equally likely 

using LinearAlgebra, Printf, Statistics, BenchmarkTools, LaTeXStrings, Polynomials, PrettyTables
#plotlyjs()  #
using SparseArrays, SuiteSparse,ToeplitzMatrices
using OrdinaryDiffEq
using SummationByPartsOperators
using ForwardDiff # for derivatives
using Plots; pythonplot()
using Trixi, Random

# the flux are the parts of the PDE that are differentiated with respect to the spatial variable
function flux(Q_vec::Vector,γ) # this function always evaluates the flux in the current cell
    ρ = Q_vec[1]
    m = Q_vec[2] 
    E = Q_vec[3]

    u = m / ρ
    p = (γ - 1) * ( E - 0.5 * ρ * u^2)
    F = SVector(m, ρ * u^2 + p, (E + p) * u) # SVector faster than a normal vector for small sizes
    #F = [m, ρ * u^2 + p, (E + p) * u ]
    return F
end

# for DGSEM: get polynomials, M and D from the Trixi package
function create_basis(polydeg:: Integer)
    basis = LobattoLegendreBasis(polydeg)
    M = Diagonal(basis.weights) # creates diagonal matrix
    return basis, M, basis.derivative_matrix # basis has weights and nodes (basis.weights / basis.nodes), type SVector
    # nodes are the collocation points on the interval [-1,1]
end

function set_up_mesh(basis, elements:: Integer, n_vars:: Integer) # n_vars: number of variables.
    nodes = length(basis.nodes) # number of nodes per element (polydeg + 1)

    X = zeros(nodes, elements) # X_i_l: i-th node in the l-th element, mapped GLL nodes in an array x

    dx = 1.0 / elements # cell length for the Jacobi scaling (how long an element is), in both cases length 1

    for l in 1:elements
        # left boundary of element l 
        x_l = 0.0 + (l - 1) * dx # both cases start at 0.0 here

        # mapping the reference nodes basis.nodes (ξ_i) onto the physical nodes X_i_l
        # x(ξ) = x_l + J * (1 + ξ), where J = dx/2 is half the cell length (p.75 in the notes).
        X[:,l] = x_l .+ (dx/2.0) .* (1.0 .+ basis.nodes)
    end

    return X, dx
end 

function maximum_velocity(γ, Q_left::Vector, Q_right::Vector) # velocity for the LLF flux
    eps_safe = 1e-12  # important for the roots and u to avoid an error
    ρ_left = Q_left[1]
    m_left = Q_left[2] 
    E_left = Q_left[3]

    u_left = m_left / max(eps_safe,ρ_left)
    p_left = (γ - 1) * ( E_left - 0.5 * ρ_left * u_left^2)

    ρ_right = Q_right[1]
    m_right = Q_right[2] 
    E_right = Q_right[3]

    u_right = m_right / ρ_right
    p_right = (γ - 1) * ( E_right - 0.5 * ρ_right * u_right^2)

    c_left = sqrt((γ*max(eps_safe,p_left)) / max(eps_safe,ρ_left)) # c: sound speed, always positive, c = sqrt(γ * p / σ)
    c_right = sqrt((γ*max(eps_safe,p_right)) / max(eps_safe,ρ_right))

    s = max(abs(u_left) + c_left, abs(u_right) + c_right) # the maximum absolute value of the eigenvalues of matrix Q (?)
    return s
end

# LLF flux is: 𝑓_num(𝑢−, 𝑢+) = 0.5 * (𝑓(𝑢−) + 𝑓(𝑢+)) − 0.5 * λ * (𝑢+ − 𝑢−)
function flux_llf(Q_left, Q_right, γ)
    flux_left = flux(Q_left, γ)
    flux_right = flux(Q_right, γ)
    s = maximum_velocity(γ, Q_left, Q_right)

    return 0.5 * (flux_left + flux_right) - 0.5 * s * (Q_right - Q_left)
end

function initialization(X, type_random, type_boundary, ω) # for the initial condition 

    if type_random == "Anfangswert" # use the correct ω
        ω_lokal = ω
    else
        ω_lokal = 0.5
    end

    n_vars = 3
    L = 1.0
    nodes, elements = size(X)
    Q = zeros(n_vars, nodes, elements)
    t_end = 0.0 #! scope variables, Julia complains when variables are declared inside loops
    γ = 0.0

    # loop over each element and each node
    if type_random == "Randwert"
        t_end = 1.0
        γ = 1.4

        for l in 1:elements
            for i in 1:nodes
                x = X[i, l]
                if x < 0.5
                    ρ, u, p = 1.0, 2.0, 1.0
                else 
                    ρ, u, p = 0.7, 1.0, 0.7
                end

                Q[1,i,l] =  ρ
                Q[2,i,l] =  ρ * u        
                Q[3,i,l] = p / (γ-1) + 0.5 * ρ * u^2
            end
        end
    else
        for l in 1:elements
            for i in 1:nodes
                x = X[i, l] # use X for initialization; the node position points are stored there

                if type_boundary == "periodic"
                    t_end = 0.5
                    if type_random == "γ"
                        γ = 1.1 + 0.5 * ω_lokal  
                    else
                        γ = 1.2
                    end
                    u = 1.0
                    p = 10.0
                    ρ = 1.0 + exp(-20 * (x - L*ω_lokal)^2)
                
                elseif type_boundary == "neumann"
                    t_end = 0.2
                    if type_random == "γ"
                        γ = 1.1 + 0.5 * ω_lokal
                    else
                        γ = 1.4
                    end

                    if x < ω
                        ρ, u, p = 1.0, 0.0, 1.0
                    else 
                        ρ, u, p = 0.125, 0.0, 0.1
                    end
                end

                Q[1,i,l] =  ρ
                Q[2,i,l] =  ρ * u        
                Q[3,i,l] = p / (γ-1) + 0.5 * ρ * u^2
            end
        end
    end
    return Q, t_end, γ
end

# right-hand side of the ODE (method of lines)
# in-place function: dU/dt = rhs!(dU,U,p,t)
function rhs!(dQ,Q,p,t,ω)
    dQ .= 0.0 # important: set the time derivative to zero before each pass

    # unpack parameters
    basis, M, D, dx, γ, type_random, type_boundary, type_boundary_value, ω = p
    weights = basis.weights
    inv_D_T_M = D' * M # for the calculation of the volume term 

    # dimensions
    n_vars, nodes, elements = size(Q) # size returns the dimensions of a matrix

    # temporary storage for the numerical fluxes at the boundaries
    F_num = zeros(n_vars, 2, elements)

    # 1. compute the numerical fluxes F_num, we need them at the boundaries of the elements (and for the LLF flux)
    for l in 1:elements
        #! flux at the right boundary of element l (interface l -> l+1)
        Q_left = Q[:,nodes,l]

        # determine Q_right (left side of l+1)
        if l < elements
            # internal interface: l -> l+1
            Q_right = Q[:,1, l+1]
        else    
            if type_boundary == "periodic"
                # periodic boundary condition: last element (l=N_e) -> first element (l=1)
                Q_right = Q[:,1, 1]
            elseif type_boundary == "neumann" || type_random == "Randwert" # Q_left and Q_right are the same here (gas flows on without resistance/no wall)
                Q_right = Q[:,nodes, l]
            end
        end
        # store the flux at the right boundary of l (F_num[2,l])
        F_num[:,2,l] = flux_llf(Q_left, Q_right, γ) # two: right door of the element

        #! flux at the left boundary of element l (interface l-1 -> l)
        Q_right = Q[:,1, l] # Q_right (left side of l)

        # determine Q_left (right side of l-1)
        if l > 1
            # internal interface: l-1 -> l
            Q_left = Q[:,nodes, l-1]
        else
            if type_random == "Randwert"
                ρ = 0.0
                p = 1.0 + exp(-3.0 * ω * t)
                u = 2.0

                if type_boundary_value == "variiert" 
                    ρ = 1.0 + exp(-3.0 * ω * t)
                elseif type_boundary_value == "konstant"
                    ρ = 1.0
                end

                E = p / (γ-1) + 0.5 * ρ * u^2
                m = u * ρ

                Q_left = [ρ, m, E]              
            else
                if type_boundary == "periodic"
                    # periodic boundary condition: first element (l=1) -> last element (1=N_e)
                    Q_left = Q[:,nodes, elements]
                elseif type_boundary == "neumann"
                    Q_left = Q[:,1,1]
                end
            end
        end
        # store the flux at the left boundary of l (F_num[1, l])
        F_num[:,1,l] = flux_llf(Q_left, Q_right, γ)
    end

    # 2. compute the discrete time derivative term dU
    for l in 1:elements
        # compute the physical flux f(U) at all nodes of the element, #! nodal basis
        flux_val = zeros(n_vars, nodes)
        for i in 1:nodes
            flux_val[:, i] = flux(Q[:, i, l], γ)
        end

        for v in 1:n_vars # formula S.78 in the notes, dQ is changed in-place directly
            # volume term: M⁻¹ * Dᵀ * M * f(U)
            # note: inv(M) is 1/weights
            vol = (1.0 ./ weights) .* (inv_D_T_M * flux_val[v, :])
            
            # surface term: M⁻¹ * Rᵀ * B * F_num
            # only at the first and last node
            surf_1 = (1.0 / weights[1]) * F_num[v, 1, l] # 1/w_1 * f_num_links
            surf_p = (1.0 / weights[nodes]) * F_num[v, 2, l] # - 1/w_p * f_num_rechts

            # assemble for each variable v
            # fill all nodes of element l with the result from the volume term
            dQ[v, :, l] .= (2.0 / dx) .* vol
            # update for the boundary nodes now
            dQ[v, 1, l]     += (2.0 / dx) * surf_1
            dQ[v, nodes, l] -= (2.0 / dx) * surf_p
        end
    end

    return dQ
end

function maximum_velocity_DG(Q_node, γ) # check the maximum wave speed at each node
    ρ, m, E = Q_node[1], Q_node[2], Q_node[3]
    u = m / ρ
    p = (γ - 1) * (E - 0.5 * ρ * u^2)
    c = sqrt(γ * max(1e-10, p) / max(1e-10, ρ)) # ensure no zero values occur
    return abs(u) + c
end

# Haupt-Solver-Funktion
function solve_dgsem(polydeg:: Integer, elements:: Integer, type_random, type_boundary, type_boundary_value, var_idx::Integer, ω)

    # 1. SetUp
    n_vars = 3
    basis, M, D = create_basis(polydeg)
    X, dx = set_up_mesh(basis, elements, n_vars)

    # 2. Initalisierung
    Q0, t_end, γ = initialization(X, type_random, type_boundary, ω)


    history = []
    save_times = []

    #! manual calculation without ODEProblem solver
    Q = copy(Q0)
    t = 0.0
    cfl_parameter = 0.1

    dQ = zeros(size(Q)) # initialize dQ
    nodes = polydeg + 1 # state how many nodes there are

    while t < t_end
        max_s = 0.0
        # Wir prüfen alle Knoten in allen Elementen
        for l in 1:elements
            for i in 1:nodes
                # helper function for sound speed
                s = maximum_velocity_DG(Q[:, i, l], γ)
                max_s = max(max_s, s)
            end
        end

        # 2. CFL condition / divide by 2*polydeg + 1
        dt = cfl_parameter * dx / ((2 * polydeg + 1) * max_s)
        
        if t + dt > t_end
            dt = t_end - t
        end

        # use rhs! to compute the next Q
        parameters = (basis, M, D, dx, γ, type_random, type_boundary,type_boundary_value, ω) # define parameter tuple
        rhs!(dQ, Q, parameters, t, ω)
        
        # explicit Euler update: Q_new = Q_old + dt * dQ/dt
        Q .+= dt .* dQ

        t += dt

        if t % 0.005 < dt  # sparse saving
            # we copy the current state of the requested variable (var_idx)
            # and flatten it [nodes * elements]
            push!(history, vec(copy(Q[var_idx, :, :]))) 
            push!(save_times, t)
        end
    end

    return Q, X, dx, history, save_times # Q is the final state here   
end

function constant_maker(omega, all_omegas, y_data:: Vector) # all omegas over one grid point
    difference = abs(omega - all_omegas[1]) # compare omega with the first entry of y_data
    value = y_data[1]
    n = length(all_omegas)

    for i in 2:n
        new_difference = abs(omega - all_omegas[i])

        if new_difference < difference 
           difference = new_difference
           value = y_data[i]
        end  
    end

    return value  # input an omega value and get a corresponding value
end

function cubic_maker(omega, all_omegas, y_data:: Vector) 
    N = length(all_omegas)

    dl = zeros(N-1) # lower subdiagonal
    d = zeros(N) # main diagonal
    du = zeros(N-1) # upper subdiagonal

    d[1] = 3.0 # to satisfy the Neumann condition 
    d[N] = 3.0

    # build the large matrix
    for i in 2:N-1
        dl[i-1] = all_omegas[i+1]- all_omegas[i] # last element is zero due to boundary condition
        d[i] = 2*(all_omegas[i+1]- all_omegas[i-1])
        du[i] = all_omegas[i]- all_omegas[i-1] # first element is zero due to boundary condition
    end

    A = Tridiagonal(dl, d, du)

    # build the right-hand side
    b = zeros(N)
    for i in 2:N-1
        dx_1 = all_omegas[i] - all_omegas[i-1] # for the weightings in the formula for b
        dx_2 = all_omegas[i+1] - all_omegas[i]
        b[i] = 3 * ((y_data[i+1] - y_data[i]) * (dx_1 / dx_2) + (y_data[i] - y_data[i-1]) * (dx_2 / dx_1))
    end

    # assemble and solve the linear system
    delta_val = A \ b # slopes are computed
    # to save the polynomials
    kubisch_splines = Vector{Polynomial{Float64}}(undef,N-1) #undef: uninitialized entries,
    # N-1 polynomials because there are N-1 intervals
    
    for i in 1:N-1
        dx = all_omegas[i+1] - all_omegas[i] # an interval length
        c_0 = y_data[i] # formula from the notes for c_0, c_1, c_2, c_3
        c_1 = delta_val[i]
        c_2 = (1.0 / (dx^2)) * (3*y_data[i+1] - 3*y_data[i] - 2*delta_val[i] *dx - delta_val[i+1] * dx)
        c_3 = (1.0 / (dx^3)) * (-2*y_data[i+1] + 2*y_data[i] + delta_val[i]*dx + delta_val[i+1]*dx)
        
        # implement (x-x_i)^k
        x = Polynomial([0.0,1.0]) # creates the polynomial x (Polynomial goes from lowest to highest power)
        # 0.0 * x^0 + 1.0 * x^1
        s_i = c_0 + c_1*(x-all_omegas[i]) + c_2*(x-all_omegas[i])^2 + c_3*(x-all_omegas[i])^3
        kubisch_splines[i] = s_i
    end
    #! # cell centers are chosen for all_omegas. For N=10, that would be [0.05, 0.15, 0.25, ..., 0.95]
    i = searchsortedlast(all_omegas,omega) # search the interval [w[i], w[i+1]] that contains ω.

    if i < 1
        i = 1          # if omega lies all the way on the left (< 0.05), use the first interval. Example for N = 10
    elseif i >= N
        i = N - 1      # if omega lies all the way on the right (> 0.95), use the last interval. Example for N = 10
    end

    # plug omega into the i-th polynomial to obtain the value
    ws_eval = kubisch_splines[i](omega) # apply the corresponding spline to ω
    return ws_eval
end

function polynom_maker(omega, all_omegas, y_data:: Vector)

    omega_mapped = 2.0 * omega - 1.0 # map omega to the reference interval [-1,1]
    omega_mapped = clamp(omega_mapped, -1.0, 1.0) # clamp(value, lower bound, upper bound)

    IM = polynomial_interpolation_matrix(basis.nodes, [omega_mapped]) # 1xN matrix

    ws_eval_vector = IM * y_data # vector output is produced, but it is only a scalar

    return ws_eval_vector[1]
end

function complete_simulation(type_random, type_boundary, type_boundary_value) # pass both the type of the random variable and the boundary condition, as well as the boundary condition of the second task
    # n is now the number of ELEMENTS
    # we fix p in order to observe convergence over n
    ns = [2,4,8,16,32,64,128]
    polydeg = 3
    nodes = polydeg + 1
    results = [] 
    var_idx = 1 # 1 for ρ, 2 for m, 3 for E
    var_names = ["Density_ρ", "Impuls_m", "Energy_E"]
    current_var_name = var_names[var_idx]

    all_results_per_n = [] # for each n, maybe still to be used
    results_dict = Dict{Int, Matrix{Float64}}() # optionally also for saving
    
    for n in ns # try all ω for each n
        n_points = (polydeg+1) * n # how many grid points there are
        dx = 1.0 / n
        dim = n_points
        # stochastic part
        all_omegas_equi = range(dx/2, 1.0 - dx/2, length=n) #! couple the number of omegas to the number of elements
        results_matrix_for_n_equi = zeros(n_points, n) # a result block for all ω per n

        for i in 1:n
            ω = all_omegas_equi[i]
            Q_end, X, history, save_times = solve_dgsem(polydeg, n, type_random, type_boundary, type_boundary_value, var_idx, ω)

            q_final = vec(Q_end[var_idx, :, :]) # for the EOC and mesh convergence
            results_matrix_for_n_equi[:, i] = q_final # each column is for one omega, each row is for one grid point
        end

        # stochastic collocation method
        polydeg_stoch = n - 1 # create a large polynomial that matches the number of all_omegas
        # important for polynomial evaluation
        basis_haupt = LobattoLegendreBasis(polydeg_stoch) # zeros of the derivative of Legendre polynomials as nodes
        #! important: polynomial evaluation does not use equidistant omegas, but the omegas at the nodes
        #! of the Legendre polynomials
        all_omegas_lobatto = 0.5 .* (basis_haupt.nodes .+ 1.0) # shift to [0,1]
        results_lobatto = zeros(n_points, n)

        for i in 1:n
            ω = all_omegas_lobatto[i]
            Q_end, _, _, _ = solve_dgsem(polydeg, n, type_random, type_boundary, type_boundary_value, var_idx, ω)
            results_lobatto[:, i] = vec(Q_end[var_idx, :, :])
        end

        for j in 1:dim
            y_data_equi = results_matrix_for_n_equi[j,:] # the data from one row
            # piecewise constant functions
            interpol_constant = constant_maker(omega, all_omegas_equi, y_data_equi)
            # kubische Splines
            interpol_cubic = cubic_maker(omega, all_omegas_equi, y_data_equi)
            # Polynome (GLL)
            y_data_lobatto = results_lobatto[j, :]
            interpol_polynom = polynom_maker(omega, basis_haupt, y_data_lobatto)
        end
    end
end

function main()
    println("Welche Zufallsvariabele-Simulation soll gestartet werden?")
    println("1: Die Anfangswerte hängen von ω ab")
    println("2: γ hängt von ω ab")
    println("3: Die Randwerte hängen von ω ab")

    auswahl_1 = readline()
    
    # determine the variable for the type of random variable
    type_random = ""
    type_boundary_value = "standard"

    if auswahl_1 == "1"
        type_random = "Anfangswert"
    elseif auswahl_1 == "2"
        type_random = "γ"
    elseif auswahl_1 == "3"
        type_random = "Randwert"

        println("Wählen Sie eine der beiden aus: ")
        println("1. Randwertbedingung (ρ variiert)")
        println("2. Randwertbedingung (ρ konstant)")

        auswahl_3 = readline()
        if auswahl_3 == "1"
            type_boundary_value = "variiert"
        elseif auswahl_3 == "2"
            type_boundary_value = "konstant"
        else
            println("Ungültige Eingabe: Bitte 1 oder 2 wählen")
            return
        end
    else
        println("Ungültige Eingabe! Bitte 1, 2 oder 3 wählen.")
        return 
    end

    println("Welche Simulation soll gestartet werden?")
    println("A: Periodische Randbedingungen")
    println("B: Neumann Randbedingung")
    
    auswahl_2 = uppercase(readline()) # converts a to A automatically

    type_boundary = ""
    if auswahl_2 == "A"
        type_boundary = "periodic"
    elseif auswahl_2 == "B"
        type_boundary = "neumann"
    else
        println("Ungültige Eingabe! Bitte A oder B wählen.")
        return 
    end

    complete_simulation(type_random, type_boundary, type_boundary_value)
end
