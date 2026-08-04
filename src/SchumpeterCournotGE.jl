"""
    SchumpeterCournotGE

A Schumpeterian growth model with Cournot competition and endogenous
participation. Solves the firm's dynamic research problem, the symmetric
policy equilibrium among rivals, and the aggregate general equilibrium.

# Getting started

```julia
using SchumpeterCournotGE

res = run_model(n = 2, η̄ = 2.0, γ = 1.06)   # build, solve, simulate, report
plot_results(res)                            # eight panels
plot_results(res; save = "run.png")

res.model.sol.aggs        # the equilibrium (g_w, g_y, ŷ)
res.panel                 # the industry cross-section
res.sym_panel             # industries that started level
```

# The pieces, innermost first

| module | what it does |
|:---|:---|
| `SymStateArrays` | storage over states whose rivals are exchangeable |
| `StateGrids` | grids, simplicial interpolation, affine extrapolation |
| `DSICModel` | `Params`, `Settings`, `Solution`, `DSIC`, save/load |
| `StaticMarket` | the within-industry Cournot equilibrium, in closed form |
| `ResearchPolicyFOC` | the innovation hazard and the optimal research effort |
| `ValueIteration` | loop 1 — the firm's problem given rivals and aggregates |
| `SymPolicyEquilibrium` | loop 2 — rivals' policy as a fixed point |
| `GeneralEquilibrium` | loop 3 — the aggregates as a fixed point |
| `Runner` | build, solve, simulate, report |
| `Plotting` | pictures (the only part that needs `Plots`) |

Each submodule's names are re-exported here, so `using SchumpeterCournotGE`
is enough. Reach for a submodule by name only when two of them would
collide.

# Three numbers worth watching

  * `contraction_modulus(params, aggs)` — must be below one, or the firm's
    problem has no bounded solution and the value iteration diverges
    rather than failing.
  * `outside_frac` in the trace — the share of simulated firms whose policy
    was *extrapolated* past the grid. A few per cent is what the
    extrapolation is for; a large share means the policy is invented.
  * the two `aᵢ/ã` lines in [`report`](@ref) — the same statistic from two
    different starting cross-sections. They should agree; if not, `burnin`
    is too short.
"""
module SchumpeterCournotGE

# --- infrastructure ---------------------------------------------------
include("core/SymStateArray.jl")
using .SymStateArrays

include("core/StateGrid.jl")
using .StateGrids

include("core/DSICModel.jl")
using .DSICModel

include("utils/ProgressBar.jl")
using .ProgressBars

# --- economics --------------------------------------------------------
include("economics/StaticMarket.jl")
using .StaticMarket

include("economics/ResearchPolicyFOC.jl")
using .ResearchPolicyFOC

# --- solvers ----------------------------------------------------------
include("solver/VFI.jl")
using .ValueIteration

include("solver/SymPolicyEq.jl")
using .SymPolicyEquilibrium

include("solver/GeneralEquilibrium.jl")
using .GeneralEquilibrium

# --- analysis ---------------------------------------------------------
include("analysis/Runner.jl")
using .Runner

include("analysis/Simulation.jl")
using .Simulation

# `Plots` dominates load time and nothing else needs it. Comment this pair
# out to use the package headless; everything but `plot_results` still works.
include("analysis/Plotting.jl")
using .Plotting

# =====================================================================
#  Re-export
#
#  Forward every submodule's public names, so `using SchumpeterCournotGE`
#  is enough and callers never have to know which submodule a function
#  came from. A name exported by two submodules would be ambiguous at the
#  point of use, so the loop reports the clash at load time rather than
#  letting it surface later as a confusing MethodError.
# =====================================================================

const _SUBMODULES = (SymStateArrays, StateGrids, DSICModel, ProgressBars,
                     StaticMarket, ResearchPolicyFOC, ValueIteration,
                     SymPolicyEquilibrium, GeneralEquilibrium, Runner,
                     Simulation, Plotting)

let seen = Dict{Symbol,Module}()
    for m in _SUBMODULES, nm in names(m)
        nm === nameof(m) && continue
        if haskey(seen, nm)
            @warn "name exported by two submodules; qualify it to use it" name = nm
        else
            seen[nm] = m
            @eval export $nm
        end
    end
end

end # module