function plot_heatmap_rho(
    testcase::EulerTestCase,
    M_vector::Vector{Int},
    ansatz_space::String,
    n::Int,
    nsnapshots::Int,
    omega_fine::Vector{Float64},
    figdir::String)

    gr()
    mkpath(figdir)

    # Physical cell centers
    dx = testcase.L / n
    x = [(i - 0.5) * dx for i in 1:n]

    for M in M_vector

        println("Running M = $M")

        # ------------------------------------------------
        # Stochastic solve + reconstruction via main
        # ------------------------------------------------
        fine_stoch = main(n, M, testcase, omega_fine, ansatz_space)

        times = fine_stoch.solutions[1].times
        nt    = length(times)

        snap_idx = unique(round.(Int, range(1, nt, length = nsnapshots)))

        # ================================================
        # Snapshot heatmaps
        # ================================================
        for j in snap_idx

            Z = zeros(length(omega_fine), n)

            for (k, solω) in enumerate(fine_stoch.solutions)
                Z[k, :] .= solω.U[j][1, :]
            end

            p = heatmap(
                x, omega_fine, Z;
                xlabel       = "x",
                ylabel       = "ω",
                title        = "ρ(x,ω), t=$(round(times[j], digits=4)), M=$M",
                colorbar     = true,
                aspect_ratio = :auto)

            savefig(p, joinpath(figdir, @sprintf("rho_M%d_snap%03d.png", M, j)))
        end

        # ================================================
        # Animated GIF
        # ================================================
        anim = @animate for j in 1:nt

            Z = zeros(length(omega_fine), n)

            for (k, solω) in enumerate(fine_stoch.solutions)
                Z[k, :] .= solω.U[j][1, :]
            end

            heatmap(
                x, omega_fine, Z;
                xlabel       = "x",
                ylabel       = "ω",
                title        = "ρ(x,ω), t=$(round(times[j], digits=4)), M=$M",
                colorbar     = true,
                aspect_ratio = :auto)
        end

        gif(anim, joinpath(figdir, @sprintf("rho_M%d_%s.gif", M, ansatz_space)), fps = 6)

        println("Saved outputs for M = $M in $figdir")
    end

    return nothing
end