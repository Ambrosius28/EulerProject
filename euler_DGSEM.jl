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

#################################################################################################
#! stochastic variable part
# different variables are now random variables, they depend on the value ω, which can take values between 0 and 1.
# an outer loop is created around the code, where the individual collocation points are processed.
# the individual solutions are then welded together into one solution by piecewise constant functions, cubic splines and polynomials
# for splines it is best to take equally spaced values of ω; for polynomials use GLL nodes.
# once the initial values should depend on ω, then the adiabatic coefficient γ
# for the adiabatic coefficient: every value of ω between 0 and 1 is equally likely 

using LinearAlgebra, Printf, Statistics, BenchmarkTools, LaTeXStrings, Polynomials, PrettyTables
#plotlyjs()  #
using SparseArrays, SuiteSparse,ToeplitzMatrices
using OrdinaryDiffEq
using SummationByPartsOperators
using ForwardDiff # for derivatives
using Plots; pythonplot()
using Trixi

# The flux are the parts of the PDE that are differentiated with respect to the spatial variable
function flux(Q_vec::Vector,gamma) # this function always evaluates the flux in the respective cell
    ρ = Q_vec[1]
    m = Q_vec[2] 
    E = Q_vec[3]

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

function maximum_velocity(gamma, Q_left::Vector, Q_right::Vector) # velocity for the LLF flux
    eps_safe = 1e-12  # important for the square roots and u to avoid an error
    ρ_left = Q_left[1]
    m_left = Q_left[2] 
    E_left = Q_left[3]

    u_left = m_left / max(eps_safe,ρ_left)
    p_left = (gamma - 1) * ( E_left - 0.5 * ρ_left * u_left^2)

    ρ_right = Q_right[1]
    m_right = Q_right[2] 
    E_right = Q_right[3]

    u_right = m_right / ρ_right
    p_right = (gamma - 1) * ( E_right - 0.5 * ρ_right * u_right^2)

    c_left = sqrt((gamma*max(eps_safe,p_left)) / max(eps_safe,ρ_left)) # c: sound speed, always positive, c = sqrt(γ * p / σ)
    c_right = sqrt((gamma*max(eps_safe,p_right)) / max(eps_safe,ρ_right))

    s = max(abs(u_left) + c_left, abs(u_right) + c_right) # the maximum absolute value of the eigenvalues of matrix Q (?)
    return s
end

# LLF flux is: 𝑓_num(𝑢−, 𝑢+) = 0.5 * (𝑓(𝑢−) + 𝑓(𝑢+)) − 0.5 * 𝜆 * (𝑢+ − 𝑢−)
function flux_llf(Q_left, Q_right, gamma)
    flux_left = flux(Q_left, gamma)
    flux_right = flux(Q_right, gamma)
    s = maximum_velocity(gamma, Q_left, Q_right)

    return 0.5 * (flux_left + flux_right) - 0.5 * s * (Q_right - Q_left)
end

function initialization(X, type) # for the initial condition
    n_vars = 3
    L = 1.0
    nodes, elements = size(X)
    Q = zeros(n_vars, nodes, elements) # initialization where the solutions of the components are written
    t_end = 0.0 #! scope variables, Julia complains if you declare variables inside loops
    gamma = 0.0

    # loop over each element and each node
    for l in 1:elements
        for i in 1:nodes
            x = X[i, l] # use X for initialization, the node position points are stored there

            if type == "periodic"
                t_end = 0.5
                gamma = 1.2
                u = 1.0
                p = 10.0
                ρ = 1.0 + exp(-20 * (x - L/2)^2)
            
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

            Q[1,i,l] =  ρ
            Q[2,i,l] =  ρ * u        
            Q[3,i,l] = p / (gamma-1) + 0.5 * ρ * u^2
        end
    end
    return Q, t_end, gamma
end

# Right-hand side of the ODE (method of lines)
# In-place function: dU/dt = rhs!(dU,U,p,t)
function rhs!(dQ,Q,p,t)
    dQ .= 0.0 # important: set the time derivative to zero before each pass

    # unpack parameters
    basis, M, D, dx, gamma, type = p
    weights = basis.weights
    inv_D_T_M = D' * M # for the computation of the volume term

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
            if type == "periodic"
                # periodic boundary condition: last element (l=N_e) -> first element (l=1)
                Q_right = Q[:,1, 1]
            elseif type == "neumann" # Q_left and Q_right are equal here (gas flows on without resistance/no wall)
                Q_right = Q[:,nodes, l]
            elseif type == "test"
                Q_right = Q[:,nodes, l]
            end
        end
        # store the flux at the right boundary of l (F_num[2,l])
        F_num[:,2,l] = flux_llf(Q_left, Q_right, gamma) # two: right door of the element

        #! flux at the left boundary of element l (interface l-1 -> l)
        Q_right = Q[:,1, l] # Q_right (left side of l)

        # determine Q_left (right side of l-1)
        if l > 1
            # internal interface: l-1 -> l
            Q_left = Q[:,nodes, l-1]
        else
            if type == "periodic"
                # periodic boundary condition: first element (l=1) -> last element (1=N_e)
                Q_left = Q[:,nodes, elements]
            elseif type == "neumann"
                Q_left = Q[:,1,1]
            elseif type == "test"
                Q_left = Q[:,1,1]
            end
        end
        # store the flux at the left boundary of l (F_num[1, l])
        F_num[:,1,l] = flux_llf(Q_left, Q_right, gamma)
    end

    # 2. compute the discrete time derivative term dU
    for l in 1:elements
        # compute the physical flux f(U) at all nodes of the element, #! nodal basis
        flux_val = zeros(n_vars, nodes)
        for i in 1:nodes
            flux_val[:, i] = flux(Q[:, i, l], gamma)
        end

        for v in 1:n_vars # formula S.78 in the notes, dQ is changed in-place directly
            # volume term: M⁻¹ * Dᵀ * M * f(U)
            # note: inv(M) is 1/weights
            vol = (1.0 ./ weights) .* (inv_D_T_M * flux_val[v, :])
            
            # surface term: M⁻¹ * Rᵀ * B * F_num
            # only at the first and last node
            surf_1 = (1.0 / weights[1]) * F_num[v, 1, l] # 1/w_1 * f_num_left
            surf_p = (1.0 / weights[nodes]) * F_num[v, 2, l] # - 1/w_p * f_num_right

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

function maximum_velocity_DG(Q_node, gamma) # look at every node for the largest velocity
    ρ, m, E = Q_node[1], Q_node[2], Q_node[3]
    u = m / ρ
    p = (gamma - 1) * (E - 0.5 * ρ * u^2)
    c = sqrt(gamma * max(1e-10, p) / max(1e-10, ρ)) # ensure that no zero values occur
    return abs(u) + c
end

# Haupt-Solver-Funktion
function solve_dgsem(polydeg:: Integer, elements:: Integer, type, var_idx::Integer)

    # 1. setup
    n_vars = 3
    basis, M, D = create_basis(polydeg)
    X, dx = set_up_mesh(basis, elements, n_vars)

    # 2. initialization
    Q0, t_end, gamma = initialization(X, type)
    history = []
    history_animation = []
    save_times = []
    x_axis = vec(X) # extract x values from the grid matrix X
    #! if it looks strange: try sortperm

    ρ = vec(Q0[1,:,:])
    m = vec(Q0[2,:,:])
    u = m ./ ρ # for the velocity
    p1_anfang = plot(x_axis, ρ, title="Density ρ", xlabel="x", ylabel="kg/m³")
    p2_anfang = plot(x_axis, m, title="Momentum m", xlabel="x", ylabel="kg/(m²s)")
    p3_anfang = plot(x_axis, vec(Q0[3,:,:]), title="Energy E", xlabel="x", ylabel="J/m³")
    p4_anfang = plot(x_axis, u, title="Velocity u", xlabel="x", ylabel="m/s", color=:red)

    display(plot(p1_anfang, p2_anfang, p3_anfang, p4_anfang, layout=(4,1), plot_title = "Simulation n = $elements",size=(800, 1100)))
    savefig("plots_dgsem/$(type)_dgsem_anfangsbedingung_n_$elements.png")

    #! manual computation without ODEProblem solver
    Q = copy(Q0)
    t = 0.0
    cfl_parameter = 0.1

    dQ = zeros(size(Q)) # initialize dQ
    nodes = polydeg + 1 # indicate how many nodes there are

    while t < t_end
        max_s = 0.0
        # we check all nodes in all elements
        for l in 1:elements
            for i in 1:nodes
                # helper function for sound speed
                s = maximum_velocity_DG(Q[:, i, l], gamma)
                max_s = max(max_s, s)
            end
        end

        # 2. CFL condition / divide by 2*polydeg + 1
        dt = cfl_parameter * dx / ((2 * polydeg + 1) * max_s)
        
        if t + dt > t_end
            dt = t_end - t
        end

        #! for animation
        ######################################################################################
        if t == 0.0 || (t % 0.01 < dt)  # saves approximately every 0.01 time units + the initial value
            current_ρ = copy(Q[1,:,:])
            current_m = copy(Q[2,:,:])
            current_E = copy(Q[3,:,:])
            current_u = current_m ./ current_ρ
                
            # save current arrays as a tuple
            push!(history_animation, (current_ρ, current_m, current_E, current_u, t))
        end

        # use rhs! to compute the next Q
        parameters = (basis, M, D, dx, gamma, type) # define parameter tuple
        rhs!(dQ, Q, parameters, t)
        
        # explicit Euler update: Q_new = Q_old + dt * dQ/dt
        Q .+= dt .* dQ

        t += dt

        if t % 0.005 < dt  # sparse saving
            # we copy the current state of the desired variable (var_idx)
            # and flatten it [nodes * elements]
            push!(history, vec(copy(Q[var_idx, :, :]))) 
            push!(save_times, t)
        end
    end
    ###############################################################################
    # animation
    println("Creating animation for n = $elements...")

    anim = @animate for step in history_animation
        # unpack the data for the current time step
        _ρ, _m, _E, _u, _t = step
            
        # format the current time for the title (2 decimal places)
        t_str = @sprintf("%.3f", _t) 
            
        # create individual plots (important: fix ylims so the axes do not jump!)
        # note: replace y_min and y_max with sensible values for your initial condition
        # if the axes wobble too much in the GIF.
        p1_anim = plot(x_axis, vec(_ρ), title="Density ρ", xlabel="x", ylabel="kg/m³", legend=false)
        p2_anim = plot(x_axis, vec(_m), title="Momentum m", xlabel="x", ylabel="kg/(m²s)", legend=false)
        p3_anim = plot(x_axis, vec(_E), title="Energy E", xlabel="x", ylabel="J/m³", legend=false)
        p4_anim = plot(x_axis, vec(_u), title="Velocity u", xlabel="x", ylabel="m/s", color=:red, legend=false)
            
        # combine in a 4x1 layout (analogous to your final images)
        plot(p1_anim, p2_anim, p3_anim, p4_anim, 
            layout=(4,1), 
            plot_title="Simulation n = $elements |  Time t = $t_str s",
            plot_titlefontsize=14, 
            size=(800, 1000))
    end

    gif(anim, "plots_dgsem/$(type)_simulation_n_$(elements).gif", fps=8) # save as GIF
    ##################################################################################

    return Q, X, dx, history, save_times # Q is the final state here
    
    #######################################################################################################
    #! only in case you want to use the ODE solver package 
    # parameter tuple for the ODEProblem
    #=parameters = (basis, M, D, dx, gamma, type)

    # 3. set up the ODE problem
    tspan = (0.0, t_end)
    prob = ODEProblem(rhs!, Q0, tspan, parameters)

    # 4. solve the ODE
    # use a robust explicit Runge-Kutta method (RK4)
    # the step size dt=0.01 is a guess and must be determined for actual stability tests
    # based on the CFL condition.
    sol = solve(prob, RK4(),dt=0.01,saveat = 0.05)=#
    #######################################################################################################

    #return sol, X, dx
end

function complete_simulation(type)
    # n is now the number of elements
    # we fix p to observe convergence over n
    ns = [2,4,8,16,32,64,128]
    polydeg = 
    nodes = polydeg + 1
    results = [] 
    var_idx = 1 # 1 for ρ, 2 for m, 3 for E
    var_names = ["Density_ρ", "Impuls_m", "Energy_E"]
    current_var_name = var_names[var_idx]
    
    for n in ns
        Q_end, X, dx, history, save_times = solve_dgsem(polydeg, n, type, var_idx)
        
        # 2. extract data
        # sol.u[end] is a 3D array [variable, node, element]
        #Q_end = sol.u[end] # only important when using the ODE solver

        q_final = vec(Q_end[var_idx, :, :]) # for the EOC and mesh convergence
        push!(results, q_final)

        #! plotting, flatten so it can be plotted
        x_flat = vec(X)
        perm = sortperm(x_flat)
        
        ρ = vec(Q_end[1,:,:])
        m = vec(Q_end[2,:,:])
        u = m ./ ρ # for the velocity
        p1 = plot(x_flat[perm], ρ[perm], title="Density ρ",xlabel="x", ylabel="kg/m³")
        p2 = plot(x_flat[perm], m[perm], title="Impuls m",xlabel="x", ylabel="kg/(m²s)")
        p3 = plot(x_flat[perm], vec(Q_end[3, :, :])[perm], title="Energy E", xlabel="x", ylabel="J/m³")
        p4 = plot(x_flat[perm], u[perm], title="Velocity u", xlabel="x", ylabel="m/s", color=:red)
        display(plot(p1, p2, p3,p4, layout=(4,1), plot_title="DGSEM n=$n, p=$polydeg",size=(800, 1100)))
        savefig("plots_dgsem/$(type)_dgsem_euler_n_$n.png")

        #! Heatmap
        x_axis_sorted = x_flat[perm]

        # the data in history must be sorted according to the x-axis
        # if you pushed vec(Q) into history:
        plot_data = reduce(hcat, history)'  # matrix: [time, location]
        plot_data_sorted = plot_data[:, perm]

        heatmap(x_axis_sorted, save_times, plot_data_sorted, 
                title="Time evolution of ($current_var_name)",
                xlabel="Location x", ylabel="Time t",
                color=:magma)
        savefig("plots_dgsem/$(type)_dgsem_heatmap_$(current_var_name)_n_$(n)_.png")
    end



    #! first try without EOC/convergence study 
    
    # EOC:
    #
    
    errors = Float64[]
    for i in 1:length(ns)-1
        q_coarse = results[i] # vector of length (polydeg+1) * ns[i]
        q_fine = results[i+1] # vector of length (polydeg+1) * ns[i+1]
        
        # the average of the nodes is computed for each element
        m_coarse = [mean(q_coarse[(e-1)*nodes + 1 : e*nodes]) for e in 1:ns[i]]
        # here the average of the nodes on the finer grid is computed for each element
        m_fine_raw = [mean(q_fine[(e-1)*nodes + 1 : e*nodes]) for e in 1:ns[i+1]]
        # average of two neighboring elements on the finer grid
        m_fine_avg = [(m_fine_raw[j*2-1] + m_fine_raw[j*2]) / 2 for j in 1:length(m_coarse)]
        
        # L1 norm: mean absolute error: sum(abs(error)) * dx 
        err = sum(abs.(m_coarse .- m_fine_avg)) / length(m_coarse) # can be divided by ns[i], since L = 1.0 in both cases
        # L2 norm: sqrt(sum(abs(error)^2) * dx)
        #err = sqrt(sum(abs.(m_coarse .- m_fine_avg).^2) / length(m_coarse)) # length(m_coarse) the same as ns[i]
        # Linf norm: max(abs(error))
        #err = maximum(abs.(m_coarse .- m_fine_avg))
        println("Error between n=$(ns[i]) and n=$(ns[i+1]): ", err)
        push!(errors, err)
    end

    # EOC calculation now remains the same as for the LLF flux
    eocs = []
    for i in 1:(length(errors)-1)
        push!(eocs, log2(errors[i] / errors[i+1]))
    end

    println("n values: ", ns[1:end-1])
    println("errors:  ", round.(errors, digits=6))
    println("EOCs:    ", [NaN; round.(eocs, digits=3)]) # first n has no order yet

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
#! test case 2 (Neumann condition)
# polydeg = 2, for n=2 I get a rather jagged solution curve at the end.
# polydeg = 1, for n=2 AND n=4 AND n=8 I get jagged solution curves, at n=16 the laptop freezes
# polydeg = 3, for all n no solution curve, only an empty image
# heatmaps: for polydeg 1,2,3 maybe take a heatmap of one variable (the density)
