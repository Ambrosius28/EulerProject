using Plots
using Printf
using FilePathsBase

"""
    plot_heatmap_rho(testcase, M_vector, ansatz_space, n, nsnapshots, omega_fine, figdir)

For each M:
- runs stochastic collocation (common dt)
- reconstructs solution on omega_fine
- saves:
    (1) heatmaps at nsnapshots times
    (2) GIF over ALL times
"""
function plot_heatmap_rho(
    testcase::EulerTestCase,
    M_vector::Vector{Int},
    ansatz_space::String,
    n::Int,
    nsnapshots::Int,
    omega_fine::Vector{Float64},
    figdir::String)

    gr() # it tells Plots.jl: “use GR to draw all plots from now on.”

    # ------------------------------------------------------------
    # create output directory
    # ------------------------------------------------------------
    mkpath(figdir)

    dx = testcase.L / n
    x = [(i - 1.5) * dx for i in 2:n+1] 
    if ansatz_space == "polynomial"
        nodes_type = "lobatto"
    else
        nodes_type = "uniform"
    end

    for M in M_vector

        println("Running M = $M")

        # ------------------------------------------------------------
        # 1. stochastic solve
        # ------------------------------------------------------------
       

        stochastic = stochastic_collocation_driver_common_dt(
            n,
            testcase,
            M;
            nodes_type = nodes_type
        )

        # ------------------------------------------------------------
        # 2. reconstruction on fine omega grid
        # ------------------------------------------------------------
        fine_stoch = reconstruct_stochastic_solution(
            omega_fine,
            stochastic,
            ansatz_space
        )

        times = fine_stoch.solutions[1].times
        nt = length(times)

        # snapshot indices (ONLY for static plots)
        snap_idx = round.(Int, range(1, nt, length = nsnapshots))

        # ------------------------------------------------------------
        # 3. SAVE SNAPSHOT HEATMAPS (static)
        # ------------------------------------------------------------
        for j in snap_idx

            Z = zeros(length(omega_fine), n + 2)

            for (k, solω) in enumerate(fine_stoch.solutions)
                U = solω.U[j]
                Z[k, :] .= U[1, :]   # rho
            end

            p = heatmap(
                x,
                omega_fine,
                Z;
                xlabel = "x",
                ylabel = "omega",
                title = "rho(x, ω) | t=$(round(times[j], digits=4)), M=$M",
                colorbar = true
            )

            filename = joinpath(
                figdir,
                @sprintf("rho_M%d_snap%03d.png", M, j)
            )

            savefig(p, filename)
        end

        # ------------------------------------------------------------
        # 4. GIF over ALL time steps (FULL DYNAMICS)
        # ------------------------------------------------------------
        anim = @animate for j in 1:nt

            Z = zeros(length(omega_fine), n + 2)

            for (k, solω) in enumerate(fine_stoch.solutions)
                U = solω.U[j]
                Z[k, :] .= U[1, :]
            end

            heatmap(
                x,
                omega_fine,
                Z;
                xlabel = "x",
                ylabel = "omega",
                title = "rho(x, ω) | t=$(round(times[j], digits=4)), M=$M",
                colorbar = true
            )
        end

        gif_name = joinpath(figdir, @sprintf("rho_M%d_%s.gif", M, ansatz_space))
        gif(anim, gif_name, fps = 6)

        println("Saved outputs for M=$M in $figdir")
    end

    return nothing
end