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
using Combinatorics
using Printf
using JSON3
using Dates
using Plots
using JLD2


# ============================================================
# Internal includes
# ============================================================

include("core/ModelParameters.jl")
include("core/StateIndex.jl")
include("core/ModelSettings.jl")

include("economics/Market.jl")
include("economics/Payoffs.jl")
include("economics/Innovation.jl")

include("bellman/SDF.jl")
include("bellman/Transitions.jl")
include("bellman/BellmanSystem.jl")

include("utils/ProgressBar.jl")
include("utils/PlotSaving.jl")
include("utils/SaveModel.jl")

include("solver/PolicyEvaluation.jl")
include("solver/PolicyImprovement.jl")
include("solver/PFI.jl")


# ============================================================
# Public API
# ============================================================

export ModelParameters,
       ModelEnvironment,
       ModelSettings,
       DSCIModel,
       solve_PFI!,
       solve_values!,
       update_policy!,
       save_plot,
       save_model,
       load_model

end;