# ==============================================================================
# 1D Compressible Euler Equations — Simple Galerkin / Weak-Form Method
# ==============================================================================
#
# PDE system (conservation form):
#
#   ∂t ρ + ∂x (m)         = 0    [mass conservation]
#   ∂t m + ∂x (u·m + p)   = 0    [momentum conservation]
#   ∂t E + ∂x ((E+p)·u)   = 0    [energy conservation]
#
# State vector (conserved variables):  U = (ρ, m, E)
# Primitive variables (for initial/boundary data):
#   u = m/ρ                          [velocity]
#   p = (γ-1)(E - ½ρu²)             [pressure]
#
# ==============================================================================
# WHAT IS THE GALERKIN / WEAK-FORM IDEA?
# ==============================================================================
#
# Instead of demanding that the PDE holds pointwise (classical solution),
# we multiply by a test function φ and integrate over each cell [x_{i-½}, x_{i+½}]:
#
#   ∫ φ · ∂t U dx  +  ∫ φ · ∂x F(U) dx  =  0
#
# Integrating by parts (the core of the weak/Galerkin approach):
#
#   ∫ φ · ∂t U dx  =  ∫ F(U) · ∂x φ dx  −  [φ · F_num]_{boundary}
#
# KEY IDEA: the flux at cell boundaries is replaced by a NUMERICAL FLUX F_num,
# which is shared between neighbouring cells. This couples the cells together
# and handles discontinuities (shocks) correctly.
#
# CHOICE OF BASIS / TEST FUNCTIONS:
#   We use piecewise-constant basis functions (φ = 1 on cell i, 0 elsewhere).
#   Then ∂x φ = 0 inside the cell, so the volume integral vanishes, and we get:
#
#   dx · dU_i/dt  =  −( F_num(U_i, U_{i+1}) − F_num(U_{i-1}, U_i) )
#
#   ⟹  dU_i/dt  =  −(1/dx) · ( F_num_{i+½} − F_num_{i-½} )
#
# This looks identical to the finite-volume update — and that is correct!
# The FV method IS the piecewise-constant Galerkin method (DG with p=0).
#
# The Galerkin structure is explicit in:
#   1. Using a NUMERICAL FLUX (LLF) at interfaces instead of the pointwise flux
#   2. The update comes from a WEAK FORM, not a pointwise derivative
#   3. Conservation is guaranteed by the telescoping of shared numerical fluxes
#
# TIME INTEGRATION: explicit Euler (same as original, easy to explain)
# SPATIAL FLUX:     Local Lax-Friedrichs (LLF) — also called Rusanov flux
# BOUNDARY CONDITIONS: ghost cells (same approach as original)
#
# ==============================================================================

using LinearAlgebra, Printf
using Plots; pythonplot()

# ==============================================================================
# SECTION 1 — Physical flux  F(U)
# ==============================================================================
# This is the flux vector that appears in the PDE: ∂t U + ∂x F(U) = 0
# It is evaluated pointwise at a single state vector U = (ρ, m, E).

function physical_flux(U::Vector, γ::Float64)
    ρ = U[1]   # density
    m = U[2]   # momentum density
    E = U[3]   # energy density

    u = m / ρ                          # velocity
    p = (γ - 1) * (E - 0.5 * ρ * u^2) # pressure (ideal gas relation)

    # F(U) = [m,  ρu² + p,  (E+p)u]
    return [m,  ρ * u^2 + p,  (E + p) * u]
end

# ==============================================================================
# SECTION 2 — Maximum local wave speed  s(U_left, U_right)
# ==============================================================================
# The wave speeds are the eigenvalues of the flux Jacobian ∂F/∂U:
#   λ₁ = u − c,  λ₂ = u,  λ₃ = u + c   where c = √(γp/ρ) is sound speed.
#
# We take the maximum absolute wave speed across the interface:
#   s = max( |u_L| + c_L,  |u_R| + c_R )
# This is used both for the LLF flux and for the CFL time-step condition.

function max_wave_speed(U_left::Vector, U_right::Vector, γ::Float64)
    ρ_L, m_L, E_L = U_left[1],  U_left[2],  U_left[3]
    ρ_R, m_R, E_R = U_right[1], U_right[2], U_right[3]

    u_L = m_L / ρ_L
    u_R = m_R / ρ_R

    p_L = (γ - 1) * (E_L - 0.5 * ρ_L * u_L^2)
    p_R = (γ - 1) * (E_R - 0.5 * ρ_R * u_R^2)

    c_L = sqrt(γ * p_L / ρ_L)   # speed of sound left
    c_R = sqrt(γ * p_R / ρ_R)   # speed of sound right

    return max(abs(u_L) + c_L, abs(u_R) + c_R)
end

# ==============================================================================
# SECTION 3 — Numerical flux  F_num(U_left, U_right)  [GALERKIN CORE]
# ==============================================================================
# This is the central piece of the Galerkin / weak-form method.
#
# In the weak form, the flux at the boundary between cell i and cell i+1
# cannot simply be evaluated from one side — we need a single well-defined
# value shared by both cells. This shared value is the NUMERICAL FLUX.
#
# We use the Local Lax-Friedrichs (LLF) flux, also called Rusanov flux:
#
#   F_num(U⁻, U⁺) = ½(F(U⁻) + F(U⁺)) − ½ · s · (U⁺ − U⁻)
#
# The first term:  ½(F(U⁻) + F(U⁺))  is the centred average of the flux
# The second term: ½ · s · (U⁺ − U⁻) is numerical dissipation (upwinding)
#   — it stabilises the scheme and enforces the entropy condition.
#
# This is consistent: if U⁻ = U⁺, then F_num = F(U)  (no dissipation needed).
# It is conservative: the flux leaving cell i equals the flux entering cell i+1.


function numerical_flux_llf(U_left::Vector, U_right::Vector, γ::Float64)
    F_left  = physical_flux(U_left,  γ)
    F_right = physical_flux(U_right, γ)
    s       = max_wave_speed(U_left, U_right, γ)

    # LLF formula: centred flux − dissipation term
    return 0.5 * (F_left + F_right) - 0.5 * s * (U_right - U_left)
end

# ==============================================================================
# SECTION 4 — Initial conditions
# ==============================================================================
# Initial data is given in primitive variables (ρ, u, p).
# We convert to conserved variables: m = ρ·u,  E = p/(γ-1) + ½ρu²
#
# Grid layout (same as original):
#   columns 1 and n+2 are ghost cells for boundary conditions
#   columns 2 to n+1 are the real cells (n cells total)
#   cell centre of real cell i is at x = (i - 1.5) * dx

function setup_initial_condition(n::Int, bc_type::String)
    if bc_type == "periodic"
        # Problem (i): smooth Gaussian density bump, T=0.5, γ=1.2
        L     = 1.0
        γ     = 1.2
        t_end = 0.5
        dx    = L / n

        U = zeros(3, n + 2)   # 3 state variables × (n cells + 2 ghost cells)

        for i in 2:n+1
            x = (i - 1.5) * dx        # cell centre

            # BUG FIX from original: was exp(-20*(x-(L/2)^2)), correct is:
            ρ = 1.0 + exp(-20.0 * (x - L/2)^2)
            u = 1.0
            p = 10.0

            U[1, i] = ρ
            U[2, i] = ρ * u
            U[3, i] = p / (γ - 1) + 0.5 * ρ * u^2
        end

        # Periodic ghost cells: domain wraps around
        U[:, 1]   = U[:, n+1]   # left ghost = last real cell
        U[:, n+2] = U[:, 2]     # right ghost = first real cell

        return U, t_end, dx, γ

    elseif bc_type == "neumann"
        # Problem (ii): Sod shock tube, T=0.2, γ=1.4
        L     = 1.0
        γ     = 1.4
        t_end = 0.2
        dx    = L / n

        U = zeros(3, n + 2)

        for i in 2:n+1
            x = (i - 1.5) * dx

            if x < L / 2
                ρ, u, p = 1.0,   0.0, 1.0    # left state
            else
                ρ, u, p = 0.125, 0.0, 0.1    # right state
            end

            U[1, i] = ρ
            U[2, i] = ρ * u
            U[3, i] = p / (γ - 1) + 0.5 * ρ * u^2
        end

        # Neumann ghost cells: zero-gradient extrapolation (outflow)
        U[:, 1]   = U[:, 2]     # left ghost = first real cell
        U[:, n+2] = U[:, n+1]  # right ghost = last real cell

        return U, t_end, dx, γ
    end
end

# ==============================================================================
# SECTION 5 — Boundary condition update (ghost cells)
# ==============================================================================
# Ghost cells are updated every time step before computing fluxes.
#
# Periodic:  the solution wraps around — no real boundary exists.
# Neumann:   zero-gradient condition — wave passes through without reflection.
#            Numerically: copy the last real cell into the ghost cell.
#            This means the flux difference at the boundary becomes zero,
#            which is the discrete analogue of ∂x U = 0 at the boundary.

function apply_boundary_conditions!(U::Matrix, n::Int, bc_type::String)
    if bc_type == "periodic"
        U[:, 1]   = U[:, n+1]   # left ghost  ← last real cell
        U[:, n+2] = U[:, 2]     # right ghost ← first real cell
    elseif bc_type == "neumann"
        U[:, 1]   = U[:, 2]     # left ghost  ← first real cell
        U[:, n+2] = U[:, n+1]   # right ghost ← last real cell
    end
end

# ==============================================================================
# SECTION 6 — One time step  [GALERKIN UPDATE — THE WEAK FORM IN ACTION]
# ==============================================================================
#
# This is where the Galerkin / weak-form philosophy is applied.
#
# The weak form (with piecewise-constant test functions φ_i = 1 on cell i) gives:
#
#   ∫_{cell i} φ_i · ∂t U dx = −( F_num_{i+½} − F_num_{i-½} )
#
# Since φ_i = 1 and the cell has width dx, the left side is dx · dU_i/dt.
# Dividing by dx and discretising in time with explicit Euler:
#
#   U_i^{n+1} = U_i^n − (dt/dx) · ( F_num_{i+½} − F_num_{i-½} )
#
# where F_num_{i+½} = numerical_flux_llf(U_i, U_{i+1}) is the numerical flux
# at the RIGHT edge of cell i (shared with cell i+1),
# and   F_num_{i-½} = numerical_flux_llf(U_{i-1}, U_i) at the LEFT edge.
#
# CONSERVATION: the numerical flux at each edge is the same for both
# neighbouring cells (same F_num_{i+½} leaves cell i and enters cell i+1).
# This guarantees global conservation — the Galerkin guarantee.

function galerkin_time_step(U_old::Matrix, dx::Float64, dt::Float64,
                             n::Int, γ::Float64)

    U_new = zeros(size(U_old))

    # --- Step 1: compute all interface fluxes F_num_{i+½} for i = 1…n+1 ---
    # Interface i sits between real cell i (column i+1) and cell i+1 (column i+2).
    # We store n+1 interface fluxes (interfaces between ghost–real and real–ghost too).
    interface_fluxes = zeros(3, n + 1)

    for i in 1:n+1
        # U_old[:, i]   is the LEFT state  (cell i,   which is ghost or real)
        # U_old[:, i+1] is the RIGHT state (cell i+1, which is real or ghost)
        interface_fluxes[:, i] = numerical_flux_llf(U_old[:, i], U_old[:, i+1], γ)
    end

    # --- Step 2: update each real cell using the weak-form update rule ---
    # Real cells are columns 2 to n+1.
    # For real cell i (column index i+1 in our array):
    #   F_num_{i+½} = interface_fluxes[:, i+1]   (right edge)
    #   F_num_{i-½} = interface_fluxes[:, i]     (left edge)

    for i in 2:n+1
        # Galerkin / weak-form update:
        #   U_i^{n+1} = U_i^n − (dt/dx) · ( F_num_right − F_num_left )
        U_new[:, i] = U_old[:, i] - (dt / dx) * (interface_fluxes[:, i] - interface_fluxes[:, i-1])
    end

    # Ghost cells are left as zero here; they will be reset by
    # apply_boundary_conditions! at the start of the next time step.
    return U_new
end

# ==============================================================================
# SECTION 7 — Full simulation loop with CFL time step control
# ==============================================================================
#
# CFL condition for stability:
#   dt ≤ CFL · dx / s_max
# where s_max = max over all interfaces of the local wave speed.
# This ensures no information travels more than one cell per time step.

function run_simulation(bc_type::String)
    n_values        = [32, 64, 128, 256, 512, 1024]
    cfl_parameter   = 0.8
    final_densities = []   # store final ρ for convergence study — OUTSIDE the loop

    for n in n_values
        U, t_end, dx, γ = setup_initial_condition(n, bc_type)
        t = 0.0

        density_history       = []        # snapshots of ρ over time (for heatmap)
        energy_snap_history   = []
        velocity_snap_history = []        # snapshots of u = m/ρ over time
        time_history          = Float64[]

        while t < t_end

            # Update ghost cells before computing fluxes
            apply_boundary_conditions!(U, n, bc_type)

            # --- CFL time step: scan all interfaces for maximum wave speed ---
            s_max = 0.0
            for i in 1:n+1
                s_i = max_wave_speed(U[:, i], U[:, i+1], γ)
                if s_i > s_max
                    s_max = s_i
                end
            end
            dt = cfl_parameter * dx / s_max

            # Don't overshoot the final time
            if t + dt > t_end
                dt = t_end - t
            end

            # Save snapshot of density, energy, and velocity for heatmaps/animations
            # Velocity is a primitive variable: u = m/ρ (not stored directly in U)
            push!(density_history,      copy(U[1, 2:n+1]))
            push!(energy_snap_history,  copy(U[3, 2:n+1]))
            push!(velocity_snap_history, copy(U[2, 2:n+1] ./ U[1, 2:n+1]))  # u = m/ρ
            push!(time_history, t)

            # --- Galerkin time step (the weak-form update) ---
            U = galerkin_time_step(U, dx, dt, n, γ)

            t += dt
        end

        # Store final density for convergence study
        push!(final_densities, copy(U[1, 2:n+1]))

        # --- Plots for this grid resolution ---
        x_cells = [(i - 0.5) * dx for i in 1:n]   # cell centres

        # Velocity at final time: u = m/ρ (recovered from conserved variables)
        velocity_final = U[2, 2:n+1] ./ U[1, 2:n+1]

        p1 = plot(x_cells, U[1, 2:n+1],   title="Density ρ",  xlabel="x", ylabel="ρ [kg/m³]")
        p2 = plot(x_cells, U[2, 2:n+1],   title="Momentum m", xlabel="x", ylabel="m [kg/(m²s)]")
        p3 = plot(x_cells, U[3, 2:n+1],   title="Energy E",   xlabel="x", ylabel="E [J/m³]")
        p4 = plot(x_cells, velocity_final, title="Velocity u", xlabel="x", ylabel="u [m/s]")
        display(plot(p1, p2, p3, p4, layout=(4, 1),
                     plot_title="Galerkin Euler — $(bc_type), n=$(n), T=$(t_end)"))
        savefig("galerkin_$(bc_type)_n$(n).png")

        energy_history   = reduce(hcat, energy_snap_history)'    # rows = time, cols = space
        density_matrix   = reduce(hcat, density_history)'        # rows = time, cols = space
        velocity_matrix  = reduce(hcat, velocity_snap_history)'  # rows = time, cols = space

        # Heatmap: density evolution over time
        plot_data = reduce(hcat, density_history)'
        heatmap(x_cells, time_history, plot_data,
                title="Density evolution — n=$(n)",
                xlabel="x", ylabel="t", color=:magma)
        savefig("heatmap_$(bc_type)_n$(n).png")

        # Heatmap: energy evolution
        heatmap(x_cells, time_history, energy_history,
                title="Energy evolution — n=$(n)",
                xlabel="x", ylabel="t", color=:heat)
        savefig("heatmap_energy_$(bc_type)_n$(n).png")

        # Heatmap: velocity evolution
        heatmap(x_cells, time_history, velocity_matrix,
                title="Velocity evolution — n=$(n)",
                xlabel="x", ylabel="t", color=:viridis)
        savefig("heatmap_velocity_$(bc_type)_n$(n).png")

        # Animations only for n=128 — avoids expensive GIF generation for all grids
        if n == 128

            # Animation: density
            anim_density = @animate for k in 1:length(time_history)
                plot(x_cells, density_matrix[k, :],
                     ylims  = (minimum(density_matrix) * 0.95, maximum(density_matrix) * 1.05),
                     xlabel = "x", ylabel = "ρ [kg/m³]",
                     title  = @sprintf("Density ρ — n=%d, t=%.4f", n, time_history[k]),
                     legend = false, color = :blue, lw = 2)
            end
            gif(anim_density, "anim_density_$(bc_type)_n$(n).gif", fps=30)

            # Animation: energy
            anim_energy = @animate for k in 1:length(time_history)
                plot(x_cells, energy_history[k, :],
                     ylims  = (minimum(energy_history) * 0.95, maximum(energy_history) * 1.05),
                     xlabel = "x", ylabel = "E [J/m³]",
                     title  = @sprintf("Energy E — n=%d, t=%.4f", n, time_history[k]),
                     legend = false, color = :red, lw = 2)
            end
            gif(anim_energy, "anim_energy_$(bc_type)_n$(n).gif", fps=30)

            # Animation: velocity
            anim_velocity = @animate for k in 1:length(time_history)
                plot(x_cells, velocity_matrix[k, :],
                     ylims  = (minimum(velocity_matrix) * 0.95, maximum(velocity_matrix) * 1.05),
                     xlabel = "x", ylabel = "u [m/s]",
                     title  = @sprintf("Velocity u — n=%d, t=%.4f", n, time_history[k]),
                     legend = false, color = :green, lw = 2)
            end
            gif(anim_velocity, "anim_velocity_$(bc_type)_n$(n).gif", fps=30)

        end  # end if n == 128

    end  # end for n in n_values

    # ==========================================================================
    # SECTION 8 — Convergence study (EOC)
    # ==========================================================================
    # Since no exact solution is available, we use the Cauchy criterion:
    #   error(h) ≈ ‖ρ_h − ρ_{h/2}‖_{L¹}
    #
    # To compare grids: average pairs of fine-grid cells onto the coarse grid.
    # Then compute the Experimental Order of Convergence:
    #   EOC = log2( error(h) / error(h/2) )
    # Expected: ~1 for shocks (1st order), ~1–2 for smooth problems.

    println("\n" * "="^52)
    println("  Convergence Study — Cauchy L¹ error in density ρ")
    println("="^52)
    println(@sprintf("  %-6s   %-14s   %-6s", "n", "‖ρ_h − ρ_{h/2}‖₁", "EOC"))
    println("-"^52)

    errors = Float64[]

    for i in 1:length(n_values)-1
        ρ_coarse = final_densities[i]
        ρ_fine   = final_densities[i+1]
        n_c      = n_values[i]

        # Average consecutive pairs of fine cells → same resolution as coarse
        ρ_fine_averaged = [(ρ_fine[2j-1] + ρ_fine[2j]) / 2 for j in 1:n_c]

        # L¹ error
        err = sum(abs.(ρ_coarse .- ρ_fine_averaged)) / n_c
        push!(errors, err)

        eoc_str = i == 1 ? "  —" : @sprintf("%.3f", log2(errors[i-1] / errors[i]))
        println(@sprintf("  %-6d   %-14.2e   %s", n_values[i], err, eoc_str))
    end
    println("="^52 * "\n")

    # Convergence plot
    plot(n_values[1:end-1], errors,
         xscale=:log10, yscale=:log10,
         marker=:circle, label="L¹ Cauchy error",
         xlabel="n  (number of cells)",
         ylabel="‖ρ_h − ρ_{h/2}‖₁",
         title="Convergence study — $(bc_type)")
    savefig("convergence_$(bc_type).png")

end  # end function run_simulation


# ==============================================================================
# SECTION 9 — Main entry point
# ==============================================================================

function main()
    println("1D Compressible Euler — Simple Galerkin (piecewise-constant DG)")
    println("==================================================================")
    println("Running periodic BC (smooth Gaussian, T=0.5, γ=1.2)...")
    run_simulation("periodic")
    println("Running Neumann BC (Sod shock tube, T=0.2, γ=1.4)...")
    run_simulation("neumann")
end

main()