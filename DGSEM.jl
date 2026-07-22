# one-dimensional compressible Euler equations
# ∂tσ + ∂xm = 0, describes mass conservation (m = ρ*u), ρ mass density, u velocity, m momentum density
# mass cannot simply disappear or appear from nothing
# ∂tm + ∂x (um) + ∂xp = 0, describes momentum conservation
# ∂tE + ∂x((E + p)u) = 0, describes energy conservation
# p(t,x) = (γ−1)(E(t,x) −ρ(t,x) * (u(t,x)^2/2)), γ adiabatic coefficient: how much pressure rises when compressing the
# gas without heat escaping
# E = p / (γ − 1) + 1/2 * ρ*u^2   #! p.295 RANDALL J. LEVEQUE Finite Volume Methods

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
# limiter: corrected minmod function, pp.193 - 195
# confirmation for our CFL number: p.190
# there is a CFL for L2 stability and TV (Total Variation) stability. since it is much weaker for TV,
# you can simply consult Table 2.2 on p.191 for the CFL number

using LinearAlgebra, Printf, Statistics, BenchmarkTools, LaTeXStrings, Polynomials, PrettyTables
#plotlyjs()  #
using SparseArrays, SuiteSparse,ToeplitzMatrices
using OrdinaryDiffEq
using SummationByPartsOperators
using ForwardDiff # for derivatives
using Plots; pythonplot()
using Trixi

# The flux are the parts of the PDE that are differentiated with respect to the spatial variable
function flux(U_vec::Vector,gamma) # this function always evaluates the flux in the respective cell
    ρ = U_vec[1]
    m = U_vec[2] 
    E = U_vec[3]

    u = m / ρ
    p = (gamma - 1) * ( E - 0.5 * ρ * u^2)
    F = SVector(m, ρ * u^2 + p, (E + p) * u) # SVector faster than a normal vector for small sizes
    #F = [m, ρ * u^2 + p, (E + p) * u ]
    return F
end

# for DGSEM: obtain polynomials, M and D from the Trixi package
function create_basis(polydeg:: Integer)
    basis = LobattoLegendreBasis(polydeg)
    M = Diagonal(basis.weights) # creates diagonal matrix
    return basis, M, basis.derivative_matrix # basis has weights and nodes (basis.weights / basis.nodes), type SVector
    # nodes are the collocation points on the interval [-1,1]
end

function set_up_mesh(basis, elements:: Integer, n_vars:: Integer) # n_vars: number of variables.
    nodes = length(basis.nodes) # number of nodes per element (polydeg + 1)

    X = zeros(nodes, elements) # X_i_l: i-th node in the l-th element, mapped GLL nodes in an array x

    dx = 1.0 / elements # cell length for Jacobi scaling (how long an element is), in both cases length 1

    for l in 1:elements
        # left boundary of element l 
        x_l = 0.0 + (l - 1) * dx # both cases start at 0.0 here

        # mapping the reference nodes basis.nodes (ξ_i) onto the physical nodes X_i_l
        # x(ξ) = x_l + J * (1 + ξ), where J = dx/2 is half the cell length (p.75 in the notes).
        X[:,l] = x_l .+ (dx/2.0) .* (1.0 .+ basis.nodes)
    end

    return X, dx
end 

function maximum_velocity(gamma, U_left::Vector, U_right::Vector) # velocity for the LLF flux
    eps_safe = 1e-12  # important for the square roots and u to avoid an error
    ρ_left = U_left[1]
    m_left = U_left[2] 
    E_left = U_left[3]

    u_left = m_left / max(eps_safe,ρ_left)
    p_left = (gamma - 1) * ( E_left - 0.5 * ρ_left * u_left^2)

    ρ_right = U_right[1]
    m_right = U_right[2] 
    E_right = U_right[3]

    u_right = m_right / ρ_right
    p_right = (gamma - 1) * ( E_right - 0.5 * ρ_right * u_right^2)

    c_left = sqrt((gamma*max(eps_safe,p_left)) / max(eps_safe,ρ_left)) # c: sound speed, always positive, c = sqrt(γ * p / σ)
    c_right = sqrt((gamma*max(eps_safe,p_right)) / max(eps_safe,ρ_right))

    s = max(abs(u_left) + c_left, abs(u_right) + c_right) # the maximum absolute value of the eigenvalues of matrix Q (?)
    return s
end

# LLF flux is: 𝑓_num(𝑢−, 𝑢+) = 0.5 * (𝑓(𝑢−) + 𝑓(𝑢+)) − 0.5 * 𝜆 * (𝑢+ − 𝑢−)
function flux_llf(U_left, U_right, gamma)
    flux_left = flux(U_left, gamma)
    flux_right = flux(U_right, gamma)
    s = maximum_velocity(gamma, U_left, U_right)

    return 0.5 * (flux_left + flux_right) - 0.5 * s * (U_right - U_left)
end

function initialization(X, type) # for the initial condition 
    n_vars = 3
    L = 1.0
    nodes, elements = size(X)
    U = zeros(n_vars, nodes, elements) # initialization where the solutions of the components are written
    t_end = 0.0 #! scope variables, Julia complains if you declare variables inside loops
    gamma = 0.0

    # loop over each element and each node
    for l in 1:elements
        for i in 1:nodes
            x = X[i, l] # use X for initialization, the node position points are stored theref

            if type == "periodic"
                t_end = 0.5
                gamma = 1.2
                u = 1.0
                p = 10.0
                ρ = 1.0 + exp(-80 * (x - L/2)^2)
            
            elseif type == "neumann"
                t_end = 0.2
                gamma = 1.4

                if x < L/2
                    ρ, u, p = 1.0, 0.0, 1.0
                else 
                    ρ, u, p = 0.125, 0.0, 0.1
                end
            
            elseif type == "test"
                t_end = 0.2
                gamma = 1.4

                ρ, u, p = 1.0, 0.5, 1.0
            end

            U[1,i,l] =  ρ
            U[2,i,l] =  ρ * u        
            U[3,i,l] = p / (gamma-1) + 0.5 * ρ * u^2
        end
    end
    return U, t_end, gamma
end

# Right-hand side of the ODE (method of lines)
# In-place function: dU/dt = rhs!(dU,U,p,t)
function rhs!(dU,U,p,t)
    dU .= 0.0 # important: set the time derivative to zero before each pass

    # unpack parameters
    basis, M, D, dx, gamma, type = p
    weights = basis.weights
    inv_D_T_M = D' * M # for the computation of the volume term 

    # dimensions
    n_vars, nodes, elements = size(U) # size returns the dimensions of a matrix

    # temporary storage for the numerical fluxes at the boundaries
    F_num = zeros(n_vars, 2, elements)

    # 1. compute the numerical fluxes F_num, we need them at the boundaries of the elements (and for the LLF flux)
    for l in 1:elements
        #! flux at the right boundary of element l (interface l -> l+1)
        U_left = U[:,nodes,l]

        # determine Q_right (left side of l+1)
        if l < elements
            # internal interface: l -> l+1
            U_right = U[:,1, l+1]
        else
            if type == "periodic"
                # periodic boundary condition: last element (l=N_e) -> first element (l=1)
                U_right = U[:,1, 1]
            elseif type == "neumann" # U_left and U_right are equal here (gas flows on without resistance/no wall)
                U_right = U[:,nodes, l]
            elseif type == "test"
                U_right = U[:,nodes, l]
            end
        end
        # store the flux at the right boundary of l (F_num[2,l])
        F_num[:,2,l] = flux_llf(U_left, U_right, gamma) # two: right door of the element

        #! flux at the left boundary of element l (interface l-1 -> l)
        U_right = U[:,1, l] # U_right (linke Seite von l)

        # determine U_left (right side of l-1)
        if l > 1
            # internal interface: l-1 -> l
            U_left = U[:,nodes, l-1]
        else
            if type == "periodic"
                # periodic boundary condition: first element (l=1) -> last element (1=N_e)
                U_left = U[:,nodes, elements]
            elseif type == "neumann"
                U_left = U[:,1,1]
            elseif type == "test"
                U_left = U[:,1,1]
            end
        end
        # store the flux at the left boundary of l (F_num[1, l])
        F_num[:,1,l] = flux_llf(U_left, U_right, gamma)
    end

    # 2. compute the discrete time derivative term dU
    for l in 1:elements
        # compute the physical flux f(U) at all nodes of the element, #! nodal basis
        flux_val = zeros(n_vars, nodes)
        for i in 1:nodes
            flux_val[:, i] = flux(U[:, i, l], gamma)
        end

        for v in 1:n_vars # formula S.78 in the notes, dQ is changed in-place directly
            # volume term: M⁻¹ * Dᵀ * M * f(U)
            # inv(M) is 1/weights
            vol = (1.0 ./ weights) .* (inv_D_T_M * flux_val[v, :])
            
            # Surface Term: M⁻¹ * Rᵀ * B * F_num
            # Only on first and last node
            surf_1 = (1.0 / weights[1]) * F_num[v, 1, l] # 1/w_1 * f_num_left
            surf_p = (1.0 / weights[nodes]) * F_num[v, 2, l] # - 1/w_p * f_num_right

            # assemble for each variable v
            # fill all nodes of element l with the result from the volume term
            dU[v, :, l] .= (2.0 / dx) .* vol
            # update for the boundary nodes now 
            dU[v, 1, l]     += (2.0 / dx) * surf_1
            dU[v, nodes, l] -= (2.0 / dx) * surf_p
        end
    end

    return dU
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

function apply_limiter!(U,basis,X,dx,type) # U: [n_vars, nodes, elements], X[nodes, elements]
    dimension = size(U)
    elements = dimension[3] # how many elements
    nodes = dimension[2] # how many nodes per element
    sum_weights = sum(basis.weights)
    half_length = dx / 2.0
    tol = 1e-5
    M = 0.6 # heuristaclly value for the corrected minmod

    mean_array = zeros(3, elements)

    # mean for each cell
    # $$\bar{U}_l = 1\Länge(I_l) ∫_{I_l} U(x) dx = 1\Länge(I_l) ∑_{k=0}_{n} Uk * wk
    for l in 1:elements  
        mean_array[1,l] = sum(basis.weights .* U[1,:,l]) / sum_weights
        mean_array[2,l] = sum(basis.weights .* U[2,:,l]) / sum_weights
        mean_array[3,l] = sum(basis.weights .* U[3,:,l]) / sum_weights
    end

    for l in 1:elements
        if l == 1
            ρ_mean_right = mean_array[1,l+1]; m_mean_right = mean_array[2,l+1]; E_mean_right = mean_array[3,l+1]
            if type == "periodic"
                ρ_mean_left = mean_array[1,elements]; m_mean_left = mean_array[2,elements]; E_mean_left = mean_array[3,elements]
            elseif type == "neumann"
                ρ_mean_left = mean_array[1,1]; m_mean_left = mean_array[2,1]; E_mean_left = mean_array[3,1]
            end
        elseif l == elements
            ρ_mean_left = mean_array[1,l-1]; m_mean_left = mean_array[2,l-1]; E_mean_left = mean_array[3,l-1]
            if type == "periodic"
               ρ_mean_right = mean_array[1,1]; m_mean_right = mean_array[2,1]; E_mean_right = mean_array[3,1]
            elseif type == "neumann"
                ρ_mean_right = mean_array[1,elements]; m_mean_right = mean_array[2,elements]; E_mean_right = mean_array[3,elements]
            end
        else
            ρ_mean_left = mean_array[1,l-1]; m_mean_left = mean_array[2,l-1]; E_mean_left = mean_array[3,l-1]
            ρ_mean_right = mean_array[1,l+1]; m_mean_right = mean_array[2,l+1]; E_mean_right = mean_array[3,l+1]
        end

        ρ_mean = mean_array[1,l]
        m_mean = mean_array[2,l]
        E_mean = mean_array[3,l]

        # in element value of the right boundary: v_{j+1/2}^{-}
        ρ_right_value = U[1,nodes,l]
        m_right_value = U[2,nodes,l]
        E_right_value = U[3,nodes,l]

        # in element value of the left boundary: v_{j-1/2}^{+}
        ρ_left_value = U[1,1,l]
        m_left_value = U[2,1,l]
        E_left_value = U[3,1,l]

        # for each of the three variables u_{j+1/2}^{-} and u_{j-1/2}^{+}, for the formulas (2.10) and (2.11)
        a1_right = ρ_right_value - ρ_mean
        a1_left = ρ_mean - ρ_left_value
        a2 = ρ_mean - ρ_mean_left
        a3 = ρ_mean_right - ρ_mean
        #U_ρ_right_value = ρ_mean + minmod(a1_right, a2, a3)
        #U_ρ_left_value = ρ_mean - minmod(a1_left, a2, a3)
        U_ρ_right_value = ρ_mean + minmod_corrected(a1_right, a2, a3,M,dx) # formula 2.10
        U_ρ_left_value = ρ_mean - minmod_corrected(a1_left, a2, a3,M,dx) # formula 2.11


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

        x_center_l = sum(X[:,l]) / nodes # Stützstellen bei Gauß-Legendre-Knoten symmetrisch um Mittelpunkt

        # quadratischen Fehler zwischen U und L minimieren:
        # F(a, b) = ∫{-1}^{1} (U(ξ) - (a + bξ))^2 
        # leite Ausdruck nach b ab und setze gleich Null
        # Term  ∫_{-1}^{1} ξ dξ fällt aufgrund der Orthogonalität weg
        # stelle nach b um:
        # b = ∫_{-1}^{1} U(ξ)*ξ dξ/∫_{-1}^{1} ξ^2 dξ, in L^2-Skalarproduktform:
        # ⟨ U, ξ ⟩ / ⟨ ξ, ξ ⟩, benutze Gauß-Lobatto-Quadratur zum Integrieren
        # Zähler ist ≈ ∑_{k=1}^{N} weights_k ⋅ U_k ⋅ nodes_k 
        # Nenner ist  ≈ ∑_{k=1}^{N} weights_k ⋅ (nodes_k)^2 (norm_factor im Code)
        basis_nodes = basis.nodes
        norm_factor = sum(basis.weights .* basis_nodes.^2)
        
        # Bedingung (ii), ob u_{j+1/2}^{-} = v_{j+1/2}^{-} und u_{j-1/2}^{+} = v_{j-1/2}^{+}
        # für ρ
        if (abs(U_ρ_right_value - ρ_right_value) > tol || abs(U_ρ_left_value - ρ_left_value) > tol)
           # Berechnung der L2-Projektions-Steigung für die Dichte (Variable 1)
           v_x_rho = sum(basis.weights .* U[1,:,l] .* basis_nodes) / norm_factor
           v_x_rho *= (1.0 / half_length) # Skalierungsfaktor, um in den echten Raum abzubilden
           # Jetzt den Limiter auf diese physikalisch korrekte Steigung anwenden:
           slope = minmod(v_x_rho, (ρ_mean_right - ρ_mean) / half_length, (ρ_mean - ρ_mean_left) / half_length)
           #local_slope = (U[1,nodes,l] - U[1,1,l]) / (X[nodes,l] - X[1,l])
           #slope = minmod(local_slope, (ρ_mean_right - ρ_mean) / half_length, (ρ_mean - ρ_mean_left) / half_length)
           # neues, lineares Polynom wird erzeugt
           U[1,:,l] = ρ_mean .+ (X[:,l] .- x_center_l) .* slope
        end

        # für m
        if (abs(U_m_right_value - m_right_value) > tol || abs(U_m_left_value - m_left_value) > tol)
           v_x_m = sum(basis.weights .* U[2,:,l] .* basis_nodes) / norm_factor
           v_x_m *= (1.0 / half_length)
           slope = minmod(v_x_m, (m_mean_right - m_mean) / half_length, (m_mean - m_mean_left) / half_length)
           #local_slope = (U[2,nodes,l] - U[2,1,l]) / (X[nodes,l] - X[1,l])
           #slope = minmod(local_slope, (m_mean_right - m_mean) / half_length, (m_mean - m_mean_left) / half_length)
           U[2,:,l] = m_mean .+ (X[:,l] .- x_center_l) .* slope
        end

        # für E
        if (abs(U_E_right_value - E_right_value) > tol || abs(U_E_left_value - E_left_value) > tol)
           v_x_E = sum(basis.weights .* U[3,:,l] .* basis_nodes) / norm_factor
           v_x_E *= (1.0 / half_length)
           slope = minmod(v_x_E, (E_mean_right - E_mean) / half_length, (E_mean - E_mean_left) / half_length)
           #local_slope = (U[3,nodes,l] - U[3,1,l]) / (X[nodes,l] - X[1,l])
           #slope = minmod(local_slope, (E_mean_right - E_mean) / half_length, (E_mean - E_mean_left) / half_length)
           U[3,:,l] = E_mean .+ (X[:,l] .- x_center_l) .* slope
        end
    end
    return U
end

function maximum_velocity_DG(U_node, gamma) # gucke an jedem Knoten nach der größten Geschwindigkeit 
    ρ, m, E = U_node[1], U_node[2], U_node[3]
    u = m / ρ
    p = (gamma - 1) * (E - 0.5 * ρ * u^2)
    c = sqrt(gamma * max(1e-10, p) / max(1e-10, ρ)) # sicherstellen, dass keine Nullwerte kommen
    return abs(u) + c
end

# Haupt-Solver-Funktion
function solve_dgsem(polydeg:: Integer, elements:: Integer, type, var_idx::Integer)

    # 1. SetUp
    n_vars = 3
    basis, M, D = create_basis(polydeg)
    X, dx = set_up_mesh(basis, elements, n_vars)

    # 2. Initalisierung
    U0, t_end, gamma = initialization(X, type)
    history = []
    history_animation = []
    save_times = []
    x_axis = vec(X) # x-Werte aus der Gittermatrix X entnehmen
    #! falls komisch aussieht: sortperm mal probieren

    ρ = vec(U0[1,:,:])
    m = vec(U0[2,:,:])
    u = m ./ ρ # für die Geschwindigkeit
    p1_anfang = plot(x_axis, ρ, title="Dichte ρ", xlabel="x", ylabel="kg/m³")
    p2_anfang = plot(x_axis, m, title="Impuls m", xlabel="x", ylabel="kg/(m²s)")
    p3_anfang = plot(x_axis, vec(U0[3,:,:]), title="Energie E", xlabel="x", ylabel="J/m³")
    p4_anfang = plot(x_axis, u, title="Geschwindigkeit u", xlabel="x", ylabel="m/s", color=:red)

    display(plot(p1_anfang, p2_anfang, p3_anfang, p4_anfang, layout=(4,1), plot_title = "Simulation n = $elements",size=(800, 1100)))
    savefig("plots_dgsem/$(type)_dgsem_anfangsbedingung_n_$elements.png")

    #! manuelle Rechnung ohne ODEProblem Solver
    U = copy(U0)
    t = 0.0
    cfl_parameter = 0.1

    dU = zeros(size(U)) # dU initalisieren
    nodes = polydeg + 1 # sagen, wie viele nodes es gibt

    while t < t_end
        max_s = 0.0
        # Wir prüfen alle Knoten in allen Elementen
        for l in 1:elements
            for i in 1:nodes
                # Hilfsfunktion für Schallgeschwindigkeit
                s = maximum_velocity_DG(U[:, i, l], gamma)
                max_s = max(max_s, s)
            end
        end

        # 2. CFL-Bedingung/ durch 2*polydeg + 1 dividieren
        dt = cfl_parameter * dx / ((2 * polydeg + 1) * max_s)
        
        if t + dt > t_end
            dt = t_end - t
        end

        #! für Animation
        ######################################################################################
        if t == 0.0 || (t % 0.01 < dt)  # Speichert ca. alle 0.01 Zeiteinheiten + den Startwert
            current_ρ = copy(U[1,:,:])
            current_m = copy(U[2,:,:])
            current_E = copy(U[3,:,:])
            current_u = current_m ./ current_ρ
                
            # speicher aktuelle Arrays als Tuple ab
            push!(history_animation, (current_ρ, current_m, current_E, current_u, t))
        end
        #####################################################################################

        #! RKDG Method: S.196 Shu, für den zeitlichen Teil zuständig, nicht den räumlichen
        # nutze rhs!, um das nächste U zu berechnen
        parameters = (basis, M, D, dx, gamma, type) # Parameter Tupel definieren

        U_n = copy(U) # U_n ist U(0) für die RK-Stufen, U wird in jeder Stufe überschrieben

        # Stufe 1
        rhs!(dU, U, parameters, t)
        #U .+= dt .* dU# expliziter Euler-Schritt

        
        U .= U_n + dt .* dU
        #if type == "neumann"
        apply_limiter!(U, basis, X, dx,type) # in jeder Zwischenstufe muss der Limiter angewandt werden
        #end

        # Stufe 2
        rhs!(dU, U, parameters, t + dt) # t übergeben eigentlich nicht wichtig bei uns, da autonomes Problem 
        U .= 0.75 * U_n + 0.25 .* (U + dt .* dU)
        #if type == "neumann"
        apply_limiter!(U, basis, X, dx,type)
        #end

        # Stufe 3
        rhs!(dU, U, parameters, t + 0.5 * dt)
        U .= (1/3) .* U_n  + (2/3) .* (U + dt .* dU)
        #if type == "neumann"
        apply_limiter!(U, basis, X, dx,type)
        #end
        # der einzelne RK-Schritt ist nach den drei Stufen vorbei
        

        t += dt

        if t % 0.005 < dt  # Sparsames Speichern
            # Wir kopieren den aktuellen Zustand der gewünschten Variable (var_idx)
            # und klopfen ihn flach [nodes * elements]
            push!(history, vec(copy(U[var_idx, :, :]))) 
            push!(save_times, t)
        end
    end
    ###############################################################################
    # Animation
    println("Erstelle Animation für n = $elements...")

    anim = @animate for step in history_animation
        # Entpacken der Daten für den aktuellen Zeitschritt
        _ρ, _m, _E, _u, _t = step
            
        # Formatierung der aktuellen Zeit für den Titel (2 Nachkommastellen)
        t_str = @sprintf("%.3f", _t) 
            
        # Einzelplots erstellen (Wichtig: ylims fixieren, damit die Achsen nicht springen!)
        # Hinweis: Ersetze y_min und y_max mit sinnvollen Werten deiner Anfangsbedingung,
        # falls die Achsen im GIF zu sehr "wobbeln".
        p1_anim = plot(x_axis, vec(_ρ), title="Density ρ", xlabel="x", ylabel="kg/m³", legend=false)
        p2_anim = plot(x_axis, vec(_m), title="Impuls m", xlabel="x", ylabel="kg/(m²s)", legend=false)
        p3_anim = plot(x_axis, vec(_E), title="Energy E", xlabel="x", ylabel="J/m³", legend=false)
        p4_anim = plot(x_axis, vec(_u), title="Velocity u", xlabel="x", ylabel="m/s", color=:red, legend=false)
            
        # Zusammenfügen im 4x1 Layout (analog zu deinen Endbildern)
        plot(p1_anim, p2_anim, p3_anim, p4_anim, 
            layout=(4,1), 
            plot_title="Simulation n = $elements |  Time t = $t_str s",
            plot_titlefontsize=14, 
            size=(800, 1000))
    end

    gif(anim, "plots_dgsem/$(type)_simulation_n_$(elements).gif", fps=8) # als GIF abspeichern
    ##################################################################################

    return U, X, dx, history, save_times, basis.weights # U ist hier der Endzustand
    
    #######################################################################################################
    #! nur im Falle, dass man das Paket ODE Solver nutzen will 
    # Parameter-Tupel für die ODEProblem
    #=parameters = (basis, M, D, dx, gamma, type)

    # 3. ODE-Problem aufstellen
    tspan = (0.0, t_end)
    prob = ODEProblem(rhs!, U0, tspan, parameters)

    # 4. Lösen der ODE
    # Verwendung eines robusten expliziten Runge-Kutta-Verfahrens (RK4)
    # Die Schrittweite dt=0.01 ist eine Vermutung und muss für tatsächliche Stabilitätstests 
    # über die CFL-Bedingung bestimmt werden.
    sol = solve(prob, RK4(),dt=0.01,saveat = 0.05)=#
    #######################################################################################################

    #return sol, X, dx
end

function complete_simulation(type)
    # n ist jetzt die Anzahl der ELEMENTE
    # Wir fixieren p, um die Konvergenz über n zu sehen
    ns = [2,4,8,16,32,64,128,256]
    polydeg = 2 
    nodes = polydeg + 1
    results = [] 
    var_idx = 2 # 1 für ρ, 2 für m, 3 für E
    var_names = ["Density_ρ", "Impuls_m", "Energy_E"]
    current_var_name = var_names[var_idx]

    #_, _, _, _, _, weights = solve_dgsem(polydeg, ns[1], type, var_idx)

    X_array = []
    
    for n in ns
        U_end, X, dx, history, save_times, _ = solve_dgsem(polydeg, n, type, var_idx)
        
        # 2. Daten extrahieren
        # sol.u[end] ist ein 3D Array [Variable, Knoten, Element]
        #U_end = sol.u[end] # nur beim ODE Solver verwenden wichtig 

        u_final = vec(U_end[var_idx, :, :]) # für den EOC und die mesh konvergenz
        push!(results, u_final)

        #! Plotten, flachklopfen, damit man plotten kann 
        x_flat = vec(X)
        push!(X_array, x_flat)
        perm = sortperm(x_flat)
        
        ρ = vec(U_end[1,:,:])
        m = vec(U_end[2,:,:])
        u = m ./ ρ # für die Geschwindigkeit
        p1 = plot(x_flat[perm], ρ[perm], title="Density ρ",xlabel="x", ylabel="kg/m³")
        p2 = plot(x_flat[perm], m[perm], title="Impuls m",xlabel="x", ylabel="kg/(m²s)")
        p3 = plot(x_flat[perm], vec(U_end[3, :, :])[perm], title="Energy E", xlabel="x", ylabel="J/m³")
        p4 = plot(x_flat[perm], u[perm], title="Velocity u", xlabel="x", ylabel="m/s", color=:red)
        display(plot(p1, p2, p3,p4, layout=(4,1), plot_title="DGSEM n=$n, p=$polydeg",size=(800, 1100)))
        savefig("plots_dgsem/$(type)_dgsem_euler_n_$n.png")

        #! Heatmap
        x_axis_sorted = x_flat[perm]

        # Die Daten in der History müssen entsprechend der x-Achse sortiert werden
        # Falls du vec(U) in die history gepusht hast:
        plot_data = reduce(hcat, history)'  # Matrix: [Zeit, Ort]
        plot_data_sorted = plot_data[:, perm]

        heatmap(x_axis_sorted, save_times, plot_data_sorted, 
                title="Time evolution of ($current_var_name)",
                xlabel="Location x", ylabel="Time t",
                color=:magma)
        savefig("plots_dgsem/$(type)_dgsem_heatmap_$(current_var_name)_n_$(n)_.png")
    end

    #! erstmal ohne EOC/Konvergenzstudio ausprobieren 
    
    # EOC:
    
    gb = GaussLegendre(polydeg)
    Gauss_nodes = gb.nodes # polydeg + 1 Vektor
    Gauss_weights = gb.weights # polydeg + 1 Vektor

    errors = Float64[]

    for i in 1:length(ns)-1
        err = 0.0
        for e in 1:ns[i+1] # go through all cells of the finer grid

            h = 1.0 / ns[i+1] # Cell width on the true interval

            u_fine = results[i+1][(e-1)*nodes + 1 : e*nodes] # Data of the current fine cell
            #u_coarse = results[i][(e-1)*nodes + 1 : e*nodes] # Data of the current coarse cell

            nodes_fine = X_array[i+1][(e-1)*nodes + 1 : e*nodes] # nodes of the current fine cell

            u_fine_gauss = interpolate(Gauss_nodes, u_fine, LobattoLegendre(polydeg)) # vector with polydeg + 1 elements

            # which coarse cell has the fine cell?
            center_fine = sum(nodes_fine) / length(nodes_fine) # middle point of the current fine cell
            # go through all coarse cells (for e in 1:ns[i]). argmin returns the index of the coarse cell, where the middle
            # point is the closest to the middle point of the fine cell
            #ratio = ns[i+1] / ns[i]
            #e_coarse = ceil(Int, e / ratio)
            e_coarse = argmin([abs(sum(X_array[i][(e-1)*nodes + 1 : e*nodes])/nodes - center_fine) for e in 1:ns[i]])

            #!
            # is center fine in the boundaries of the croase cell
            x_min_coarse = minimum(X_array[i][(e_coarse-1)*nodes + 1 : e_coarse*nodes])
            x_max_coarse = maximum(X_array[i][(e_coarse-1)*nodes + 1 : e_coarse*nodes])

            if !(x_min_coarse <= center_fine <= x_max_coarse)
                println("ALARM: Zuteilung fehlerhaft!")
                println("Feine Zelle $e hat Zentrum $center_fine, aber grobe Zelle $e_coarse geht von $x_min_coarse bis $x_max_coarse")
            end
            #!

            #e_coarse = argmin([abs(sum(X_array[i][(e-1)*nodes + 1 : e*nodes])/nodes - center_fine) for e in 1:ns[i]])
            u_coarse_cell = results[i][(e_coarse-1)*nodes + 1 : e_coarse*nodes]

            # Where is the fine cell located in relation to the center of the coarse cell?
            # center_fine is the middle point of the fine cell
            # center_coarse is the middle point of the corresponding coarse cell
            center_coarse = sum(X_array[i][(e_coarse-1)*nodes + 1 : e_coarse*nodes]) / nodes

            if center_fine < center_coarse
                # fine cell is located on the left side
                gauss_nodes_for_coarse = (Gauss_nodes .- 1) ./ 2 # vector with polydeg + 1 elements
            else
                # fine cell is located on the right side
                gauss_nodes_for_coarse = (Gauss_nodes .+ 1) ./ 2 # vector with polydeg + 1 elements
            end

            #if i == 6
                #println("e_coarse is: ", e_coarse)
                #println("")

            # interpolation: 
            u_coarse_gauss = interpolate(gauss_nodes_for_coarse, u_coarse_cell, LobattoLegendre(polydeg))

            #err += 0.5*h*sum( Gauss_weights .* (u_coarse_gauss .- u_fine_gauss) .^2 ) # squared L2-norm
            err += 0.5 * h * sum(Gauss_weights .* abs.(u_coarse_gauss .- u_fine_gauss))
        end
        println("Fehler zwischen n=$(ns[i]) und n=$(ns[i+1]): ", err)
        push!(errors, sqrt(err))
    end
    
    # EOC Berechnung bleibt jetzt gleich zum LLF-Fluss
    eocs = []
    for i in 1:(length(errors)-1)
        push!(eocs, log2(errors[i] / errors[i+1]))
    end

    println("n-Werte: ", ns[1:end-1])
    println("Fehler:  ", round.(errors, digits=6))
    println("EOCs:    ", [NaN; round.(eocs, digits=3)]) # Erstes n hat noch keine Ordnung

    p_conv = plot(ns[2:end], errors, 
                  xscale=:log10, yscale=:log10, 
                  marker=:circle, label="L1 norm",
                  title="Mesh Convergence Study",
                  xlabel="n (number of intervals)", ylabel="||u_h - u_h/2||")
    display(p_conv)
    savefig("plots_dgsem/$(type)_dgsem_konvergenz_studie_$(current_var_name)_.png")
    
end

function main()
    println("Welche Simulation soll gestartet werden?")
    println("1: Periodische Randbedingungen")
    println("2: Neumann Randbedingung")
    println("3: Test (Richtig implementiert?)")

    auswahl = readline()
    if auswahl == "1"
        complete_simulation("periodic")
    elseif auswahl == "2"
        complete_simulation("neumann")
    elseif auswahl == "3"
        complete_simulation("test")
    else
        println("Ungültige Eingabe! Bitte 1, 2 oder 3 wählen.")
    end
end
