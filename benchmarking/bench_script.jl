# ==============================================================================
# Script for benchmarking the FV_stochastic.jl code.
# ==============================================================================
using BenchmarkTools
using CSV
using DataFrames
using TimerOutputs

include("../FV_stochastic.jl")
include("../plots.jl")

version = "5_inline"
testcase = exercise_2_2_i
figdir = "./benchmarking/bench_figures"

parameters = Parameters(
    n = 200,
    M = 32,
    nomega_fine = 100,
    ansatz_space = "constant",
    nsnapshots = 8
)

println(parameters)
println()
println(testcase)


# println("\nSimulating...\n")
# sol = main_old(testcase, parameters)   # Compile everything first
# println("\nSimulation completed.\n")
# plot_heatmap_rho_T(testcase, parameters, sol, figdir)

println("\nBenchmarking...\n")
trial = @benchmark main($testcase, $parameters) #CHANGE HERE THE SOLVER
display(trial)

# open(joinpath(figdir, "benchmark_$(version).txt"), "w") do io
#     #show(io, trial)
#     println(io, trial)
# end

# Save results to CSV
results = DataFrame(
    Version = [version],
    Function = ["main"],
    Minimum_ms = [minimum(trial).time / 1e6],
    Median_ms = [median(trial).time / 1e6],
    Mean_ms = [mean(trial).time / 1e6],
    Memory_MB = [trial.memory / 1024^2],
    Allocations = [trial.allocs],
)

CSV.write(joinpath(figdir, "benchmark_$(version).csv"), results)

reset_timer!(TO)
main(testcase, parameters) #CHANGE HERE THE SOLVER
show(TO)

open(joinpath(figdir, "timeroutput_$(version).txt"), "w") do io
    show(io, TO)
end

# # Save results to CSV
# new_result = DataFrame(
#     Version = [version],
#     Function = ["main"],
#     Minimum_ms = [minimum(trial).time / 1e6],
#     Median_ms = [median(trial).time / 1e6],
#     Mean_ms = [mean(trial).time / 1e6],
#     Memory_MB = [trial.memory / 1024^2],
#     Allocations = [trial.allocs],
# )

# filename = "results.csv"

# if isfile(filename)
#     results = CSV.read(filename, DataFrame)
#     append!(results, new_result)
# else
#     results = new_result
# end

# CSV.write(filename, results)