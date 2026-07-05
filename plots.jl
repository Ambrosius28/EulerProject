function plot_heatmap_rho(testcase::EulerTestCase, par::Parameters, figdir::String)

    M_values = par.M_values
    ansatz_space = par.ansatz_space
    n = par.n
    nsnapshots = par.nsnapshots
    omega_fine = par.omega_fine

    gr()
    mkpath(figdir)

    # Physical cell centers
    dx = testcase.L / n
    x = [(i - 0.5) * dx for i in 1:n]

    for M in M_values

        println("Running M = $M")
        parM = Parameters(; # we have to do this because doing par.M = M would not work since par is immutable
            n = par.n,
            M = M,
            M_values = par.M_values,
            omega_fine = par.omega_fine,
            ansatz_space = par.ansatz_space,
            cfl_parameter = par.cfl_parameter,
            nsnapshots = par.nsnapshots)
        # ------------------------------------------------
        # Stochastic solve + reconstruction via main
        # ------------------------------------------------
        fine_stoch = main(testcase, parM)

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

function plot_mean_rho(testcase::EulerTestCase, par::Parameters, figdir::String)

    M_values = par.M_values
    ansatz_space = par.ansatz_space
    n = par.n
    nsnapshots = par.nsnapshots
    omega_fine = par.omega_fine

    gr()
    mkpath(figdir)

    # Physical cell centers
    dx = testcase.L / n
    x = [(i - 0.5) * dx for i in 1:n]

    for M in M_values

        println("Running M = $M")
        parM = Parameters(; # we have to do this because doing par.M = M would not work since par is immutable
            n = par.n,
            M = M,
            M_values = par.M_values,
            omega_fine = par.omega_fine,
            ansatz_space = par.ansatz_space,
            cfl_parameter = par.cfl_parameter,
            nsnapshots = par.nsnapshots)
        # ------------------------------------------------
        # Stochastic solve + reconstruction via main
        # ------------------------------------------------
        fine_stoch = main(testcase, parM)

        times = fine_stoch.solutions[1].times
        nt    = length(times)

        snap_idx = unique(round.(Int, range(1, nt, length = nsnapshots)))

        # ================================================
        # Snapshot mean +/- std plots
        # ================================================
        for j in snap_idx

            Z = zeros(length(omega_fine), n)

            for (k, solω) in enumerate(fine_stoch.solutions)
                Z[k, :] .= solω.U[j][1, :]
            end

            mean_rho = vec(mean(Z, dims = 1))
            std_rho  = vec(std(Z, dims = 1))

            p = plot(
                x, mean_rho;
                ribbon       = std_rho,
                fillalpha    = 0.3,
                xlabel       = "x",
                ylabel       = "ρ",
                title        = "E[ρ](x), t=$(round(times[j], digits=4)), M=$M",
                label        = "mean ± std",
                legend       = :topright)

            savefig(p, joinpath(figdir, @sprintf("mean_rho_M%d_snap%03d.png", M, j)))
        end

        # ================================================
        # Animated GIF
        # ================================================
        anim = @animate for j in 1:nt

            Z = zeros(length(omega_fine), n)

            for (k, solω) in enumerate(fine_stoch.solutions)
                Z[k, :] .= solω.U[j][1, :]
            end

            mean_rho = vec(mean(Z, dims = 1))
            std_rho  = vec(std(Z, dims = 1))

            plot(
                x, mean_rho;
                ribbon       = std_rho,
                fillalpha    = 0.3,
                xlabel       = "x",
                ylabel       = "ρ",
                title        = "E[ρ](x), t=$(round(times[j], digits=4)), M=$M",
                label        = "mean ± std",
                legend       = :topright,
                ylims        = (minimum(mean_rho .- std_rho) - 0.1, maximum(mean_rho .+ std_rho) + 0.1))
        end

        gif(anim, joinpath(figdir, @sprintf("mean_rho_M%d_%s.gif", M, ansatz_space)), fps = 6)

        println("Saved outputs for M = $M in $figdir")
    end

    return nothing
end