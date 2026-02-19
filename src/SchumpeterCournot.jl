module SchumpeterCournot

# ============================================================
# External dependencies
# ============================================================

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Optim
using Base.Threads

# ============================================================
# Internal includes
# Order matters
# ============================================================

include("ModelParams.jl")
include("EconFuncs.jl")
include("StateIndex.jl")
include("ProgressTrack.jl")

include("Types.jl")
include("Bellman.jl")
include("Solver.jl")
include("PlotSaving.jl")

# ============================================================
# Public API
# ============================================================

export ModelSettings,
       ModelParameters,
       solve_PFI!,
       solve_values!,
       update_policy!,
       save_plot

end;