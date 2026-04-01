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

include("core/StateIndex.jl")
include("core/ScaleMode.jl")
include("core/ModelStructs.jl")

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

include("analysis/RelativePolicy.jl")

# ============================================================
# Public API
# ============================================================

        # Model Parameters and Environment
export  ModelParameters,
        ModelEnvironment,
        SolverSettings,
        SolverState,
        DSCIModel,

        # Actional Functions
        get_labour_demand,
        innovation_success_prob,

        # Value Scaling Modes
        ValueScaling,
        Levels, 
        Detrended,

        # Innovation Types
        InnovationType,
        NonChanging,
        Decreasing,

        # Solver Functions
        solve_PFI!,
        solve_values!,
        update_policy!,

        # Utilities for saving/loading and plotting
        save_plot,
        save_model,
        load_model,

        # Analysis
        policy_by_relative_state

end;