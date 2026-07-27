"""
    DSICModel

DSIC — Dynamic Stochastic Innovation and Competition. Model objects and
persistence.

# Layout

    Params      immutable   economic parameters
    Settings    immutable   numerical/solver choices, incl. the grid spec
    Solution    mutable     value, policies, convergence state
    DSIC        immutable   the three above plus the constructed StateGrid

`Params` and `Settings` are plain value objects: concretely typed, cheap to
copy, comparable with `==`, and usable as dictionary keys when sweeping
calibrations. `Solution` mutates in place across a Bellman iteration;
`DSIC` itself never does.

# Persistence

Two formats, chosen by purpose:

  * [`save_config`](@ref) / [`load_config`](@ref) — parameters and settings
    as **TOML**. Human-readable and diffable, so a calibration can be read
    and reviewed without starting Julia.
  * [`save_model`](@ref) / [`load_model`](@ref) — the whole model including
    solution arrays, as **JLD2**.

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

export Params, Settings, Solution, DSIC,
       state_length, build_grid, spacing_from, validate, reset!,
       save_config, load_config, save_model, load_model,
       to_dict, from_dict, SCHEMA_VERSION

"""
    SCHEMA_VERSION

Version of the on-disk layout. Bump it when a change cannot be absorbed by
defaults — renaming a field, or changing the meaning of one. Adding or
removing a field does not need a bump, since [`from_dict`](@ref) handles
both.
"""
const SCHEMA_VERSION = 1

# =====================================================================
#  Economic parameters
# =====================================================================

"""
    Params(; kwargs...)

Economic parameters. All fields have defaults, so `Params()` works and
`Params(β = 0.97)` changes one thing.

  * `n`  — number of firms
  * `β`  — discount factor
  * `σ`  — intertemporal substitution parameter
  * `μ`  — elasticity across industries
  * `γ`  — innovation step size
  * `θ`  — elasticity of the innovation probability w.r.t. research labour
  * `ε`  — catch-up parameter in the innovation probability
  * `η̄`  — scale parameter in the innovation probability
"""
Base.@kwdef struct Params
    n::Int      = 3
    β::Float64  = 0.96
    σ::Float64  = 2.0
    μ::Float64  = 1.5
    γ::Float64  = 1.05
    θ::Float64  = 0.50
    ε::Float64  = 0.10
    η̄::Float64  = 1.00

    function Params(n, β, σ, μ, γ, θ, ε, η̄)
        n >= 1    || throw(ArgumentError("n must be ≥ 1 (got $n)"))
        0 < β < 1 || throw(ArgumentError("β must lie in (0,1) (got $β)"))
        σ > 1     || throw(ArgumentError("σ must be > 1 (got $σ)"))
        μ > 1     || throw(ArgumentError("μ must be > 1 (got $μ)"))
        γ >= 0    || throw(ArgumentError("γ must be non-negative (got $γ)"))
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
#  Solver / simulation settings
# =====================================================================

"""
    Settings(; kwargs...)

Numerical choices: the grid specification, loop tolerances, and 
simulation controls.

Grid:

  * `gmin`, `gmax`  — range of the first state component
  * `kx`, `ky`      — nodes on the `x` axis and on the symmetric `y` axis
  * `ymin`, `ymax`  — range of the `y` axis (`NaN` ⇒ same as `gmin`/`gmax`)
  * `spacing`, `spacing_param`   — `x`-axis placement, see [`spacing_from`](@ref)
  * `yspacing`, `yspacing_param` — `y`-axis placement

Solver:

  * `outer_tol`, `outer_maxiter` — outer (e.g. equilibrium) loop
  * `inner_tol`, `inner_maxiter` — inner (e.g. Bellman) loop
  * `damping` — relaxation weight on the update, `1.0` for no damping

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
    # --- solver -------------------------------------------------------
    outer_tol::Float64      = 1e-8
    outer_maxiter::Int      = 500
    inner_tol::Float64      = 1e-10
    inner_maxiter::Int      = 1_000
    damping::Float64        = 1.0
    # --- simulation ---------------------------------------------------
    n_sims::Int             = 1_000
    n_periods::Int          = 500
    burnin::Int             = 100
    seed::Int               = 20260727
    # --- reporting ----------------------------------------------------
    verbose::Bool           = true

    function Settings(gmin, gmax, kx, ky, ymin, ymax, spacing, spacing_param,
                      yspacing, yspacing_param, outer_tol, outer_maxiter,
                      inner_tol, inner_maxiter, damping, n_sims, n_periods,
                      burnin, seed, verbose)
        gmin < gmax        || throw(ArgumentError("need gmin < gmax"))
        kx >= 2            || throw(ArgumentError("kx must be ≥ 2 (got $kx)"))
        ky >= 2            || throw(ArgumentError("ky must be ≥ 2 (got $ky)"))
        outer_tol > 0      || throw(ArgumentError("outer_tol must be positive"))
        inner_tol > 0      || throw(ArgumentError("inner_tol must be positive"))
        outer_maxiter >= 1 || throw(ArgumentError("outer_maxiter must be ≥ 1"))
        inner_maxiter >= 1 || throw(ArgumentError("inner_maxiter must be ≥ 1"))
        0 < damping <= 1   || throw(ArgumentError("damping must lie in (0,1]"))
        n_sims >= 1        || throw(ArgumentError("n_sims must be ≥ 1"))
        n_periods >= 1     || throw(ArgumentError("n_periods must be ≥ 1"))
        burnin >= 0        || throw(ArgumentError("burnin must be ≥ 0"))
        burnin < n_periods || throw(ArgumentError("burnin must be < n_periods"))
        return new(gmin, gmax, kx, ky, ymin, ymax, spacing, spacing_param,
                   yspacing, yspacing_param, outer_tol, outer_maxiter,
                   inner_tol, inner_maxiter, damping, n_sims, n_periods,
                   burnin, seed, verbose)
    end
end

"""
    spacing_from(sym::Symbol, param::Real) -> AbstractSpacing

Build a spacing from its serialisable description. Recognised:

| `sym`         | meaning                    | `param`      |
|:--------------|:---------------------------|:-------------|
| `:linear`     | `LinearSpacing()`          | ignored      |
| `:log`        | `LogSpacing()`             | ignored      |
| `:shiftedlog` | `ShiftedLogSpacing(param)` | shift `c`    |
| `:power`      | `PowerSpacing(param)`      | exponent `θ` |

For a bespoke warp, construct the `StateGrid` yourself and pass it 
to [`DSIC`](@ref).
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

# =====================================================================
#  Solution
# =====================================================================

"""
    Solution(grid, n)

Mutable state produced by the solver: the value function, the policies, and
convergence bookkeeping. Held by [`DSIC`](@ref) and updated in place, so an
`Interpolant` built over `sol.V` stays valid across iterations — update with
`copyto!(sol.V.data, new)` rather than rebinding.

Fields: `V`, `policy`, `policy_comp`, `converged`, `iters`, `residual`,
`history`.
"""
mutable struct Solution{A<:StateArray}
    V::A
    policy::A
    policy_comp::A
    converged::Bool
    iters::Int
    residual::Float64
    history::Vector{Float64}
end

Solution(grid::StateGrid, n::Integer) =
    Solution(statearray(grid, n), statearray(grid, n), statearray(grid, n),
             false, 0, Inf, Float64[])

"""
    reset!(sol::Solution)

Zero the arrays and clear the convergence record, keeping the allocations.
"""
function reset!(sol::Solution)
    fill!(sol.V, 0.0)
    fill!(sol.policy, 0.0)
    fill!(sol.policy_comp, 0.0)
    sol.converged = false
    sol.iters     = 0
    sol.residual  = Inf
    empty!(sol.history)
    return sol
end

# =====================================================================
#  DSIC
# =====================================================================

"""
    DSIC(; params = Params(), settings = Settings())
    DSIC(params, settings, grid)
    DSIC(path::AbstractString)

The whole model: parameters, settings, the grid built from them, and a
freshly allocated solution.

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
    DSIC(params, settings, grid, Solution(grid, state_length(params)))

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
              repr(s.spacing), ", outer_tol=", s.outer_tol,
              ", inner_tol=", s.inner_tol, ")")

Base.show(io::IO, m::DSIC) =
    print(io, "DSIC(n=", m.params.n, ", ", nstates(m.sol.V), " states)")

function Base.show(io::IO, ::MIME"text/plain", m::DSIC)
    println(io, "DSIC")
    println(io, "  ", m.params)
    println(io, "  ", m.settings)
    println(io, "  ", m.grid)
    print(io,   "  solution: ", nstates(m.sol.V), " states, converged=",
                m.sol.converged, ", iters=", m.sol.iters,
                ", residual=", m.sol.residual)
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
The solution is *not* written; use [`save_model`](@ref) for that.

```toml
schema = 1

[params]
n = 3
"β" = 0.96

[settings]
kx = 100
spacing = "power"
```
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

Write the complete model — parameters, settings, and the solution arrays —
to a JLD2 file.

Arrays are stored as plain `Matrix`es rather than as `StateArray`s, for the
same reason parameters are stored as dictionaries: the file then depends on
the *shape* of the data, not on the current definition of a type.
[`load_model`](@ref) reallocates from `settings` and copies the data back in.
"""
function save_model(path::AbstractString, m::DSIC)
    validate(m)
    jldopen(path, "w") do f
        f["schema"]      = SCHEMA_VERSION
        f["params"]      = to_dict(m.params)
        f["settings"]    = to_dict(m.settings)
        f["V"]           = m.sol.V.data
        f["policy"]      = m.sol.policy.data
        f["policy_comp"] = m.sol.policy_comp.data
        f["meta"]        = Dict{String,Any}(
            "converged" => m.sol.converged,
            "iters"     => m.sol.iters,
            "residual"  => m.sol.residual,
            "history"   => m.sol.history)
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

    m.sol.converged = get(meta, "converged", false)
    m.sol.iters     = get(meta, "iters", 0)
    m.sol.residual  = get(meta, "residual", Inf)
    append!(m.sol.history, get(meta, "history", Float64[]))

    validate(m)
    return m
end

end # module