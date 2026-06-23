using Plots
using Printf
using FilePathsBase

function plot_heatmap_rho(
testcase::EulerTestCase,
M_vector::Vector{Int},
ansatz_space::String,
n::Int,
nsnapshots::Int,
omega_fine::Vector{Float64},
figdir::String
)


gr()

mkpath(figdir)

# Physical cell centers only
dx = testcase.L / n
x = [(i - 1.5) * dx for i in 2:n+1]

nodes_type =
    ansatz_space == "polynomial" ? "lobatto" : "uniform"

for M in M_vector

    println("Running M = $M")

    # ----------------------------------------------------
    # Stochastic solve
    # ----------------------------------------------------
    stochastic = stochastic_collocation_driver_common_dt(
        n,
        testcase,
        M;
        nodes_type = nodes_type
    )

    # ----------------------------------------------------
    # Reconstruction on fine omega grid
    # ----------------------------------------------------
    fine_stoch = reconstruct_stochastic_solution(
        omega_fine,
        stochastic,
        ansatz_space
    )

    times = fine_stoch.solutions[1].times
    nt = length(times)

    snap_idx = unique(
        round.(Int, range(1, nt, length = nsnapshots))
    )

    # ====================================================
    # Snapshot heatmaps
    # ====================================================
    for j in snap_idx

        Z = zeros(length(omega_fine), n)

        for (k, solω) in enumerate(fine_stoch.solutions)

            U = solω.U[j]

            # Density component, physical cells only
            Z[k, :] .= U[1, 2:n+1]

        end

        p = heatmap(
            x,
            omega_fine,
            Z;
            xlabel = "x",
            ylabel = "ω",
            title = "ρ(x,ω), t=$(round(times[j], digits=4)), M=$M",
            colorbar = true,
            aspect_ratio = :auto
        )

        filename = joinpath(
            figdir,
            @sprintf("rho_M%d_snap%03d.png", M, j)
        )

        savefig(p, filename)
    end

    # ====================================================
    # Animated GIF
    # ====================================================
    anim = @animate for j in 1:nt

        Z = zeros(length(omega_fine), n)

        for (k, solω) in enumerate(fine_stoch.solutions)

            U = solω.U[j]

            # Density component, physical cells only
            Z[k, :] .= U[1, 2:n+1]

        end

        heatmap(
            x,
            omega_fine,
            Z;
            xlabel = "x",
            ylabel = "ω",
            title = "ρ(x,ω), t=$(round(times[j], digits=4)), M=$M",
            colorbar = true,
            aspect_ratio = :auto
        )
    end

    gif_name = joinpath(
        figdir,
        @sprintf("rho_M%d_%s.gif", M, ansatz_space)
    )

    gif(anim, gif_name, fps = 6)

    println("Saved outputs for M = $M in $figdir")
end

return nothing

end
