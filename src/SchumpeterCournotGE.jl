# module SchumpeterCournotGE

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

include(joinpath(@__DIR__, "..", "src", "core",     "SymStateArray.jl"));      using .SymStateArrays
include(joinpath(@__DIR__, "..", "src", "core",     "StateGrid.jl"));          using .StateGrids
include(joinpath(@__DIR__, "..", "src", "core",     "DSICModel.jl"));          using .DSICModel
include(joinpath(@__DIR__, "..", "src", "economics","StaticMarket.jl"));       using .StaticMarket
include(joinpath(@__DIR__, "..", "src", "economics","ResearchPolicyFOC.jl"));  using .ResearchPolicyFOC
include(joinpath(@__DIR__, "..", "src", "solver",   "VFI.jl"));                using .ValueIteration
include(joinpath(@__DIR__, "..", "src", "utils",    "ProgressBar.jl"));        using .ProgressBars
include(joinpath(@__DIR__, "..", "src", "economics","Monopoly.jl"));           using .Monopoly


# ============================================================
# Public API
# ============================================================

        # Model Parameters and Environment

# end