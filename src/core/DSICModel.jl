"""
    DSICModel

DSIC — Dynamic Stochastic Innovation and Competition. Model objects and
persistence.

# Layout

    Params      immutable   economic parameters (draft primitives)
    Aggregates  immutable   the iterated triple (g_w, g_y, ŷ)
    Settings    immutable   numerical/solver choices, incl. the grid spec
    LoopStatus  mutable     convergence record for one loop
    Solution    mutable     value, policies, aggregates, convergence state
    DSIC        immutable   the above plus the constructed StateGrid

`Params`, `Aggregates` and `Settings` are plain value objects: concretely
typed, cheap to copy, comparable with `==`, usable as dictionary keys when
sweeping calibrations. `Solution` mutates in place across the iteration;
`DSIC` itself never does.

# The three loops

The solution algorithm nests three fixed points, and the objects here are
sized to that structure:

 1. **VFI/PFI** — given aggregates and `policy_comp`, solve the firm's
    problem. Tolerance `tol_vfi`, recorded in `sol.vfi`.
 2. **Game** — update `policy_comp = λ_game·policy + (1-λ_game)·policy_comp`
    until `‖policy − policy_comp‖_∞ < tol_game`. Recorded in `sol.game`.
 3. **Aggregate** — simulate, recover `(g_w, g_y, ŷ)`, damp with `λ_agg`
    until the sup-norm gap is below `tol_agg`. Recorded in `sol.agg`.

Tolerances should nest, `tol_vfi < tol_game < tol_agg`, so that no loop is
chasing the numerical noise of the one inside it.

# Persistence

Two formats, chosen by purpose:

  * [`save_config`](@ref) / [`load_config`](@ref) — parameters and settings
    as **TOML**. Human-readable and diffable, so a calibration can be read
    and reviewed without starting Julia.
  * [`save_model`](@ref) / [`load_model`](@ref) — the whole model including
    solution arrays and aggregates, as **JLD2**.

Both write a `schema` version and store *field dictionaries* rather than
serialised structs. Rebuilding through the keyword constructor means a file
written today still loads after you add a field (the new field takes its
default) or remove one (the stale key is ignored). Serialising the structs
themselves would bake the current layout into every file on disk.

For the same reason `Settings` records the grid spacing as a `Symbol` plus a
parameter rather than an `AbstractSpacing`: a `WarpSpacing` holds a closure,
and closures do not serialise. The grid is reconstructed from the
specification by [`build_grid`](@ref).

# Example

```julia
p = Params(n = 3, β = 0.96, γ = 1.05)
s = Settings(kx = 100, ky = 50, spacing = :power, spacing_param = 2.0)
m = DSIC(params = p, settings = s)

m.sol.aggs                              # starting (g_w, g_y, ŷ)
save_config("calibration.toml", m)      # readable, git-friendly
save_model("run01.jld2", m)             # everything, including V

m2 = load_model("run01.jld2")
m2.params == m.params                   # true
```
"""
module DSICModel

using TOML
using JLD2

using ..SymStateArrays
using ..StateGrids

export Params, Aggregates, Settings, LoopStatus, Solution, DSIC,
       state_length, build_grid, spacing_from, validate, reset!,
       yhat_from, damp, dist, bgp_gap, converged, tolerances_nest,
       save_config, load_config, save_model, load_model,
       to_dict, from_dict, SCHEMA_VERSION

"""
    SCHEMA_VERSION

Version of the on-disk layout. Bump it when a change cannot be absorbed by
defaults — renaming a field, or changing the meaning of one. Adding or
removing a field does not need a bump, since [`from_dict`](@ref) handles
both.

`2` — split the single inner/outer tolerance pair into the three loops
(`tol_vfi`, `tol_game`, `tol_agg`) and added aggregates to `Solution`.
Version-1 files still load: their renamed keys are ignored and the new
fields take defaults.
"""
const SCHEMA_VERSION = 2

# =====================================================================
#  Economic parameters
# =====================================================================

"""
    Params(; kwargs...)

Economic parameters — the primitives of the model draft. All fields have
defaults, so `Params()` works and `Params(β = 0.97)` changes one thing.

  * `n`  — `|N(j)|` number of firms per industry
  * `β`  — household discount factor
  * `σ`  — inverse elasticity of intertemporal substitution
  * `μ`  — elasticity of substitution across industries, `μ > 1`
  * `γ`  — innovation step size, `γ > 1`
  * `θ`  — elasticity of the innovation probability w.r.t. research labour
  * `ε`  — catch-up parameter in the innovation probability
  * `η̄`  — scale parameter in the innovation probability

Invariants are checked at construction. They are deliberately loose — only
what is unambiguous — so tighten them to your model rather than discovering
a bad calibration three hours into a solve. The draft's `μ > 1` and the
step size `γ >= 1` are enforced, as is `σ > 1`; `θ`, `ε` and `η̄` are only
required to be non-negative, so `η̄ = 0` gives a no-innovation benchmark.
"""
Base.@kwdef struct Params
    n::Int      = 3
    β::Float64  = 0.96
    σ::Float64  = 2.0
    μ::Float64  = 1.5
    γ::Float64  = 1.05
    θ::Float64  = 0.30
    ε::Float64  = 0.10
    η̄::Float64  = 1.00

    function Params(n, β, σ, μ, γ, θ, ε, η̄)
        n >= 1    || throw(ArgumentError("n must be ≥ 1 (got $n)"))
        0 < β < 1 || throw(ArgumentError("β must lie in (0,1) (got $β)"))
        σ > 1     || throw(ArgumentError("σ must be > 1 (got $σ)"))
        μ > 1     || throw(ArgumentError("μ must be > 1 (got $μ)"))
        γ >= 1    || throw(ArgumentError("γ must be >= 1 (got $γ)"))
        θ >= 0    || throw(ArgumentError("θ must be non-negative (got $θ)"))
        ε >= 0    || throw(ArgumentError("ε must be non-negative (got $ε)"))
        η̄ >= 0    || throw(ArgumentError("η̄ must be non-negative (got $η̄)"))
        return new(n, β, σ, μ, γ, θ, ε, η̄)
    end
end

"""
    state_length(p::Params) -> Int

Length `n` of a state vector as `StateArray` counts it — the first
component plus the permutation-symmetric ones.

"""
state_length(p::Params) = p.n

# =====================================================================
#  Aggregates — the object the outer loop iterates on
# =====================================================================

"""
    Aggregates(; g_w = 0.0, g_y = 0.0, ŷ = 1.5)
    Aggregates(g_w, g_y, L_r, ℒ)

The aggregate quantities the firm's problem takes as given:

  * `g_w` — growth rate of the wage
  * `g_y` — growth rate of output
  * `ŷ`   — the demand shifter, `(1 - L_r) / (1 - ℒ)`

These are the outer loop's iterate: step 1 takes them as given, step 5
produces new ones, step 6 compares. `L_r` and `ℒ` are *not* stored here —
they are outputs of the simulation recorded on [`Solution`](@ref), from
which `ŷ` follows via [`yhat_from`](@ref).

On a balanced growth path `g_w == g_y`. That is not imposed: both are
iterated and the gap between them is a free diagnostic — see
[`bgp_gap`](@ref).
"""
Base.@kwdef struct Aggregates
    g_w::Float64 = 0.0
    g_y::Float64 = 0.0
    ŷ::Float64   = 1.5

    function Aggregates(g_w, g_y, ŷ)
        isfinite(g_w) || throw(ArgumentError("g_w must be finite (got $g_w)"))
        isfinite(g_y) || throw(ArgumentError("g_y must be finite (got $g_y)"))
        ŷ > 0         || throw(ArgumentError("ŷ must be positive (got $ŷ)"))
        return new(g_w, g_y, ŷ)
    end
end

"""
    yhat_from(L_r, ℒ) -> Float64

The demand shifter `ŷ = (1 - L_r) / (1 - ℒ)`, where `L_r` is the aggregate
research share of labour and `ℒ` the corresponding moment of the firm
distribution.

Throws if `ℒ == 1` (the denominator vanishes) or if the result is
non-positive, since either means the labour accounting has gone wrong
upstream and a silent `Inf` would only surface much later.
"""
function yhat_from(L_r::Real, ℒ::Real)
    ℒ == 1 && throw(ArgumentError("ℒ == 1 makes ŷ = (1-L_r)/(1-ℒ) undefined"))
    y = (1 - L_r) / (1 - ℒ)
    y > 0 || throw(ArgumentError(
        "ŷ = (1-$L_r)/(1-$ℒ) = $y is not positive"))
    return y
end

Aggregates(g_w::Real, g_y::Real, L_r::Real, ℒ::Real) =
    Aggregates(Float64(g_w), Float64(g_y), yhat_from(L_r, ℒ))

"""
    damp(new::Aggregates, old::Aggregates, λ) -> Aggregates

Relaxed update `λ·new + (1-λ)·old`, componentwise — step 6.1 of the
algorithm. `λ = 1` takes the new value outright.
"""
function damp(new::Aggregates, old::Aggregates, λ::Real)
    0 < λ <= 1 || throw(ArgumentError("λ must lie in (0,1] (got $λ)"))
    return Aggregates(λ * new.g_w + (1 - λ) * old.g_w,
                      λ * new.g_y + (1 - λ) * old.g_y,
                      λ * new.ŷ   + (1 - λ) * old.ŷ)
end

"""
    dist(a::Aggregates, b::Aggregates) -> Float64

Sup-norm gap between two aggregate vectors, the quantity compared against
`tol_agg` in step 6.
"""
dist(a::Aggregates, b::Aggregates) =
    max(abs(a.g_w - b.g_w), abs(a.g_y - b.g_y), abs(a.ŷ - b.ŷ))

"""
    bgp_gap(a::Aggregates) -> Float64

`|g_w - g_y|`. Zero on a balanced growth path. Since the two growth rates
are iterated separately rather than one being imposed on the other, this is
an independent check on the aggregation step: if it does not close,
something upstream is wrong.
"""
bgp_gap(a::Aggregates) = abs(a.g_w - a.g_y)

Base.show(io::IO, a::Aggregates) =
    print(io, "Aggregates(g_w=", a.g_w, ", g_y=", a.g_y, ", ŷ=", a.ŷ, ")")

# =====================================================================
#  Solver / simulation settings
# =====================================================================

"""
    Settings(; kwargs...)

Numerical choices: the grid specification, the three loops' tolerances, and
simulation controls. Everything here is a *number, symbol or flag* — never a
function or a constructed object — so that a `Settings` can be written to
TOML and read back.

Grid:

  * `gmin`, `gmax`  — range of the first state component
  * `kx`, `ky`      — nodes on the `x` axis and on the symmetric `y` axis
  * `ymin`, `ymax`  — range of the `y` axis (`NaN` ⇒ same as `gmin`/`gmax`)
  * `spacing`, `spacing_param`   — `x`-axis placement, see [`spacing_from`](@ref)
  * `yspacing`, `yspacing_param` — `y`-axis placement

Loops, innermost first. Tolerances should nest,
`tol_vfi < tol_game < tol_agg`:

  * `tol_vfi`,  `maxiter_vfi`               — the firm's problem
  * `tol_game`, `maxiter_game`, `λ_game`    — the `policy_comp` fixed point
  * `tol_agg`,  `maxiter_agg`,  `λ_agg`     — the aggregate fixed point

Starting point for the aggregate loop:

  * `g_w0`, `g_y0`, `ŷ0`

Simulation: `n_sims`, `n_periods`, `burnin`, `seed`.
Reporting: `verbose`.
"""
Base.@kwdef struct Settings
    # --- grid ---------------------------------------------------------
    gmin::Float64           = 0.0
    gmax::Float64           = 10.0
    kx::Int                 = 100
    ky::Int                 = 50
    ymin::Float64           = NaN          # NaN ⇒ inherit gmin
    ymax::Float64           = NaN          # NaN ⇒ inherit gmax
    spacing::Symbol         = :linear
    spacing_param::Float64  = 1.0
    yspacing::Symbol        = :linear
    yspacing_param::Float64 = 1.0
    # --- loop 1: value function ---------------------------------------
    tol_vfi::Float64        = 1e-10
    maxiter_vfi::Int        = 1_000
    # --- loop 2: policy_comp / the game -------------------------------
    tol_game::Float64       = 1e-8
    maxiter_game::Int       = 200
    λ_game::Float64         = 0.2
    # --- loop 3: aggregates -------------------------------------------
    tol_agg::Float64        = 1e-6
    maxiter_agg::Int        = 100
    λ_agg::Float64          = 0.3
    # --- starting point for loop 3 ------------------------------------
    g_w0::Float64           = 0.0
    g_y0::Float64           = 0.0
    ŷ0::Float64             = 1.5
    # --- simulation ---------------------------------------------------
    n_sims::Int             = 1_000
    n_periods::Int          = 500
    burnin::Int             = 100
    seed::Int               = 20260727
    # --- reporting ----------------------------------------------------
    verbose::Bool           = true

    function Settings(gmin, gmax, kx, ky, ymin, ymax, spacing, spacing_param,
                      yspacing, yspacing_param, tol_vfi, maxiter_vfi,
                      tol_game, maxiter_game, λ_game, tol_agg, maxiter_agg,
                      λ_agg, g_w0, g_y0, ŷ0, n_sims, n_periods, burnin,
                      seed, verbose)
        gmin < gmax       || throw(ArgumentError("need gmin < gmax"))
        kx >= 2           || throw(ArgumentError("kx must be ≥ 2 (got $kx)"))
        ky >= 2           || throw(ArgumentError("ky must be ≥ 2 (got $ky)"))
        tol_vfi > 0       || throw(ArgumentError("tol_vfi must be positive"))
        tol_game > 0      || throw(ArgumentError("tol_game must be positive"))
        tol_agg > 0       || throw(ArgumentError("tol_agg must be positive"))
        maxiter_vfi >= 1  || throw(ArgumentError("maxiter_vfi must be ≥ 1"))
        maxiter_game >= 1 || throw(ArgumentError("maxiter_game must be ≥ 1"))
        maxiter_agg >= 1  || throw(ArgumentError("maxiter_agg must be ≥ 1"))
        0 < λ_game <= 1   || throw(ArgumentError("λ_game must lie in (0,1]"))
        0 < λ_agg <= 1    || throw(ArgumentError("λ_agg must lie in (0,1]"))
        ŷ0 > 0            || throw(ArgumentError("ŷ0 must be positive"))
        n_sims >= 1       || throw(ArgumentError("n_sims must be ≥ 1"))
        n_periods >= 1    || throw(ArgumentError("n_periods must be ≥ 1"))
        burnin >= 0       || throw(ArgumentError("burnin must be ≥ 0"))
        burnin < n_periods || throw(ArgumentError("burnin must be < n_periods"))
        return new(gmin, gmax, kx, ky, ymin, ymax, spacing, spacing_param,
                   yspacing, yspacing_param, tol_vfi, maxiter_vfi, tol_game,
                   maxiter_game, λ_game, tol_agg, maxiter_agg, λ_agg,
                   g_w0, g_y0, ŷ0, n_sims, n_periods, burnin, seed, verbose)
    end
end

"""
    tolerances_nest(s::Settings) -> Bool

Whether `tol_vfi < tol_game < tol_agg`. Not enforced — there are reasons to
break it while debugging — but a loop solved less tightly than the one
inside it will stall at that noise floor rather than converge.
"""
tolerances_nest(s::Settings) = s.tol_vfi < s.tol_game < s.tol_agg

"""
    spacing_from(sym::Symbol, param::Real) -> AbstractSpacing

Build a spacing from its serialisable description. Recognised:

| `sym`         | meaning                    | `param`      |
|:--------------|:---------------------------|:-------------|
| `:linear`     | `LinearSpacing()`          | ignored      |
| `:log`        | `LogSpacing()`             | ignored      |
| `:shiftedlog` | `ShiftedLogSpacing(param)` | shift `c`    |
| `:power`      | `PowerSpacing(param)`      | exponent `θ` |

This indirection is what lets a grid survive a round trip through a file: a
`WarpSpacing` holds a closure and cannot be serialised, so `Settings` stores
the recipe and the grid is rebuilt from it. For a bespoke warp, construct
the `StateGrid` yourself and pass it to [`DSIC`](@ref).
"""
function spacing_from(sym::Symbol, param::Real)
    sym === :linear     && return LinearSpacing()
    sym === :log        && return LogSpacing()
    sym === :shiftedlog && return ShiftedLogSpacing(param)
    sym === :power      && return PowerSpacing(param)
    throw(ArgumentError("unknown spacing $(repr(sym)); expected one of " *
                        ":linear, :log, :shiftedlog, :power"))
end

"""
    build_grid(s::Settings) -> StateGrid

Construct the grid described by `s`. `ymin`/`ymax` fall back to
`gmin`/`gmax` when left as `NaN`.
"""
function build_grid(s::Settings)
    ymin = isnan(s.ymin) ? s.gmin : s.ymin
    ymax = isnan(s.ymax) ? s.gmax : s.ymax
    return StateGrid(s.gmin, s.gmax, s.kx, s.ky;
                     spacing  = spacing_from(s.spacing,  s.spacing_param),
                     yspacing = spacing_from(s.yspacing, s.yspacing_param),
                     ymin = ymin, ymax = ymax)
end

"""
    Aggregates(s::Settings) -> Aggregates

The starting point for the aggregate loop, as recorded in `s`.
"""
Aggregates(s::Settings) = Aggregates(s.g_w0, s.g_y0, s.ŷ0)

# =====================================================================
#  Convergence records
# =====================================================================

"""
    LoopStatus()

Convergence record for one loop: whether it converged, how many iterations
it took, and the residual it stopped at.
"""
mutable struct LoopStatus
    converged::Bool
    iters::Int
    residual::Float64
end

LoopStatus() = LoopStatus(false, 0, Inf)

function reset!(st::LoopStatus)
    st.converged = false
    st.iters     = 0
    st.residual  = Inf
    return st
end

Base.show(io::IO, st::LoopStatus) =
    print(io, "LoopStatus(converged=", st.converged, ", iters=", st.iters,
              ", residual=", st.residual, ")")

# =====================================================================
#  Solution
# =====================================================================

"""
    Solution(grid, n; aggs = Aggregates())

Mutable state produced by the solver. Held by [`DSIC`](@ref) and updated in
place, so an `Interpolant` built over `sol.V` stays valid across iterations
— update with `copyto!(sol.V.data, new)` rather than rebinding.

Arrays:

  * `V`           — value function
  * `policy`      — the firm's own policy, the result of loop 1
  * `policy_comp` — rivals' policy, the iterate of loop 2

Aggregates and the quantities behind them:

  * `aggs` — the current `(g_w, g_y, ŷ)`
  * `L_r`  — aggregate research share of labour, an output of the simulation
  * `ℒ`    — the corresponding moment of the distribution

`L_r` and `ℒ` are recorded rather than iterated: they are what the
simulation produces, and `ŷ` follows from them by [`yhat_from`](@ref).

Convergence, one record per loop: `vfi`, `game`, `agg`; plus `history`, the
aggregate-loop residual by iteration.
"""
mutable struct Solution{A<:StateArray}
    V::A
    policy::A
    policy_comp::A
    aggs::Aggregates
    L_r::Float64
    ℒ::Float64
    vfi::LoopStatus
    game::LoopStatus
    agg::LoopStatus
    history::Vector{Float64}
end

Solution(grid::StateGrid, n::Integer; aggs::Aggregates = Aggregates()) =
    Solution(statearray(grid, n), statearray(grid, n), statearray(grid, n),
             aggs, NaN, NaN,
             LoopStatus(), LoopStatus(), LoopStatus(), Float64[])

"""
    reset!(sol::Solution; aggs = sol.aggs)

Zero the arrays and clear every convergence record, keeping the allocations.
The aggregates are left alone by default — warm-starting the outer loop is
usually what you want; pass `aggs` to override.

Do **not** call this between iterations of the solver: consecutive problems
are nearly identical, and a warm-started inner solve takes a handful of
sweeps where a cold one takes hundreds.
"""
function reset!(sol::Solution; aggs::Aggregates = sol.aggs)
    fill!(sol.V, 0.0)
    fill!(sol.policy, 0.0)
    fill!(sol.policy_comp, 0.0)
    sol.aggs = aggs
    sol.L_r  = NaN
    sol.ℒ    = NaN
    reset!(sol.vfi); reset!(sol.game); reset!(sol.agg)
    empty!(sol.history)
    return sol
end

"""
    converged(sol::Solution) -> Bool

Whether all three loops converged. The individual records remain available
as `sol.vfi`, `sol.game`, `sol.agg` for attributing a failure.
"""
converged(sol::Solution) =
    sol.vfi.converged && sol.game.converged && sol.agg.converged

# =====================================================================
#  DSIC
# =====================================================================

"""
    DSIC(; params = Params(), settings = Settings())
    DSIC(params, settings, grid)
    DSIC(path::AbstractString)

The whole model: parameters, settings, the grid built from them, and a
solution initialised at the starting aggregates in `settings`.

`DSIC` is immutable, but `model.sol` is not — the solver writes into it.
Pass an explicit `grid` when you need a spacing that `Settings` cannot
describe (a bespoke `WarpSpacing`, say); such a model still saves, but
reloading rebuilds the grid from `settings` and will not reproduce the warp.
"""
struct DSIC{G<:StateGrid,S<:Solution}
    params::Params
    settings::Settings
    grid::G
    sol::S
end

DSIC(params::Params, settings::Settings, grid::StateGrid) =
    DSIC(params, settings, grid,
         Solution(grid, state_length(params); aggs = Aggregates(settings)))

DSIC(; params::Params = Params(), settings::Settings = Settings()) =
    DSIC(params, settings, build_grid(settings))

"""
    validate(m::DSIC)

Re-check the invariants that span objects — the ones no single constructor
can see, such as the solution arrays matching the grid.
"""
function validate(m::DSIC)
    n = state_length(m.params)
    statelength(m.sol.V) == n || throw(ArgumentError(
        "solution holds length-$(statelength(m.sol.V)) states, params imply $n"))
    gridsize(m.sol.V) == length(xaxis(m.grid)) || throw(DimensionMismatch(
        "value array and grid disagree on the number of x nodes"))
    ygridsize(m.sol.V) == length(yaxis(m.grid)) || throw(DimensionMismatch(
        "value array and grid disagree on the number of y nodes"))
    return nothing
end

Base.show(io::IO, p::Params) =
    print(io, "Params(n=", p.n, ", β=", p.β, ", σ=", p.σ, ", μ=", p.μ,
              ", γ=", p.γ, ", θ=", p.θ, ", ε=", p.ε, ", η̄=", p.η̄, ")")

Base.show(io::IO, s::Settings) =
    print(io, "Settings(kx=", s.kx, ", ky=", s.ky, ", spacing=",
              repr(s.spacing), ", tol_vfi=", s.tol_vfi, ", tol_game=",
              s.tol_game, ", tol_agg=", s.tol_agg, ")")

Base.show(io::IO, m::DSIC) =
    print(io, "DSIC(n=", m.params.n, ", ", nstates(m.sol.V), " states)")

function Base.show(io::IO, ::MIME"text/plain", m::DSIC)
    println(io, "DSIC")
    println(io, "  ", m.params)
    println(io, "  ", m.settings)
    println(io, "  ", m.grid)
    println(io, "  ", m.sol.aggs)
    println(io, "  L_r = ", m.sol.L_r, ", ℒ = ", m.sol.ℒ)
    println(io, "  vfi  ", m.sol.vfi)
    println(io, "  game ", m.sol.game)
    print(io,   "  agg  ", m.sol.agg)
end

# =====================================================================
#  Field dictionaries
#
#  Everything is written as a Dict of field name => value and rebuilt
#  through the keyword constructor. That is what makes old files readable
#  by new code: a field added since the file was written takes its default,
#  and a field since removed is ignored rather than fatal. Serialising the
#  structs themselves would freeze today's layout into every file on disk.
# =====================================================================

"""
    to_dict(x) -> Dict{String,Any}

Field name => value, for any struct.
"""
to_dict(x) = Dict{String,Any}(String(f) => getfield(x, f)
                              for f in fieldnames(typeof(x)))

_coerce(::Type{Symbol}, v) = Symbol(v)
_coerce(::Type{T}, v) where {T} = convert(T, v)

"""
    from_dict(T, d) -> T

Rebuild a `Base.@kwdef` struct from a field dictionary. Keys absent from `d`
take their defaults; keys that are not fields of `T` are ignored. Values are
converted to the declared field types, so a `Symbol` stored as a string (as
TOML requires) comes back a `Symbol`.
"""
function from_dict(::Type{T}, d::AbstractDict) where {T}
    kw = Dict{Symbol,Any}()
    for f in fieldnames(T)
        key = String(f)
        haskey(d, key) || continue
        kw[f] = _coerce(fieldtype(T, f), d[key])
    end
    return T(; kw...)
end

# TOML cannot hold a Symbol, so write symbols as strings on the way out.
_toml_value(v::Symbol) = String(v)
_toml_value(v) = v
_toml_dict(x) = Dict{String,Any}(k => _toml_value(v) for (k, v) in to_dict(x))

# =====================================================================
#  TOML: parameters and settings only
# =====================================================================

"""
    save_config(path, m::DSIC)
    save_config(path, params::Params, settings::Settings)

Write parameters and settings to `path` as TOML — human-readable and
diffable, which is what you want for a calibration under version control.
The solution is *not* written; use [`save_model`](@ref) for that. The
aggregate loop's starting point travels with the settings, so a config file
fully determines a run.
"""
function save_config(path::AbstractString, params::Params, settings::Settings)
    doc = Dict{String,Any}("schema"   => SCHEMA_VERSION,
                           "params"   => _toml_dict(params),
                           "settings" => _toml_dict(settings))
    open(path, "w") do io
        TOML.print(io, doc; sorted = true)
    end
    return path
end

save_config(path::AbstractString, m::DSIC) =
    save_config(path, m.params, m.settings)

"""
    load_config(path) -> (params, settings, schema)

Read a TOML configuration written by [`save_config`](@ref). Missing entries
fall back to defaults, so a partial file listing only the fields you care
about is a valid calibration.
"""
function load_config(path::AbstractString)
    doc    = TOML.parsefile(path)
    schema = get(doc, "schema", 0)
    p = from_dict(Params,   get(doc, "params",   Dict{String,Any}()))
    s = from_dict(Settings, get(doc, "settings", Dict{String,Any}()))
    return (p, s, schema)
end

"""
    DSIC(path::AbstractString) -> DSIC

Build a fresh, unsolved model from a TOML configuration file.
"""
function DSIC(path::AbstractString)
    p, s, _ = load_config(path)
    return DSIC(params = p, settings = s)
end

# =====================================================================
#  JLD2: the whole model
# =====================================================================

"""
    save_model(path, m::DSIC)

Write the complete model — parameters, settings, aggregates and the solution
arrays — to a JLD2 file.

Arrays are stored as plain `Matrix`es rather than as `StateArray`s, and the
solution's scalars are stored as a flat dictionary rather than as nested
structs, for the same reason parameters are: the file then depends on the
*shape* of the data, not on the current definition of a type.
[`load_model`](@ref) reallocates from `settings` and copies the data back in.
"""
function save_model(path::AbstractString, m::DSIC)
    validate(m)
    s = m.sol
    jldopen(path, "w") do f
        f["schema"]      = SCHEMA_VERSION
        f["params"]      = to_dict(m.params)
        f["settings"]    = to_dict(m.settings)
        f["V"]           = s.V.data
        f["policy"]      = s.policy.data
        f["policy_comp"] = s.policy_comp.data
        f["meta"]        = Dict{String,Any}(
            "g_w"            => s.aggs.g_w,
            "g_y"            => s.aggs.g_y,
            "yhat"           => s.aggs.ŷ,
            "L_r"            => s.L_r,
            "scriptL"        => s.ℒ,
            "vfi_converged"  => s.vfi.converged,
            "vfi_iters"      => s.vfi.iters,
            "vfi_residual"   => s.vfi.residual,
            "game_converged" => s.game.converged,
            "game_iters"     => s.game.iters,
            "game_residual"  => s.game.residual,
            "agg_converged"  => s.agg.converged,
            "agg_iters"      => s.agg.iters,
            "agg_residual"   => s.agg.residual,
            "history"        => s.history)
    end
    return path
end

"""
    load_model(path) -> DSIC

Read a model written by [`save_model`](@ref). Parameters and settings are
rebuilt through their keyword constructors, the grid is reconstructed from
the settings, fresh arrays are allocated, and the stored data is copied in.

Throws `DimensionMismatch` if the stored arrays do not match the grid the
settings describe — which is what you want, since it means the file and the
code disagree about something defaults cannot paper over.
"""
function load_model(path::AbstractString)
    schema, pd, sd, Vd, Pd, PCd, meta = jldopen(path, "r") do f
        (haskey(f, "schema") ? f["schema"] : 0,
         f["params"], f["settings"],
         f["V"], f["policy"], f["policy_comp"], f["meta"])
    end
    schema <= SCHEMA_VERSION || @warn(
        "file schema $schema is newer than this code ($SCHEMA_VERSION); " *
        "loading anyway")

    p = from_dict(Params, pd)
    s = from_dict(Settings, sd)
    m = DSIC(params = p, settings = s)

    size(Vd) == size(m.sol.V.data) || throw(DimensionMismatch(
        "stored value array is $(size(Vd)) but settings imply " *
        "$(size(m.sol.V.data))"))
    copyto!(m.sol.V.data, Vd)
    copyto!(m.sol.policy.data, Pd)
    copyto!(m.sol.policy_comp.data, PCd)

    sol = m.sol
    sol.aggs = Aggregates(get(meta, "g_w",  s.g_w0),
                          get(meta, "g_y",  s.g_y0),
                          get(meta, "yhat", s.ŷ0))
    sol.L_r = get(meta, "L_r", NaN)
    sol.ℒ   = get(meta, "scriptL", NaN)
    for (st, pre) in ((sol.vfi, "vfi"), (sol.game, "game"), (sol.agg, "agg"))
        st.converged = get(meta, pre * "_converged", false)
        st.iters     = get(meta, pre * "_iters", 0)
        st.residual  = get(meta, pre * "_residual", Inf)
    end
    append!(sol.history, get(meta, "history", Float64[]))

    validate(m)
    return m
end

end # module