using CSV
using DataFrames
using CairoMakie

results = CSV.read("results.csv", DataFrame)

main_results = filter(row -> row.Function == "main", results)

fig = Figure()
ax = Axis(fig[1,1],
    xlabel = "Version",
    ylabel = "Runtime (ms)")

barplot!(ax,
    1:nrow(main_results),
    main_results.Median_ms)

ax.xticks = (1:nrow(main_results), main_results.Version)

save("runtime.png", fig)