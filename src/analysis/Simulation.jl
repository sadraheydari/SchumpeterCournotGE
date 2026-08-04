"""
    Simulation

A standalone simulation of the economy, separate from `Runner`. Give it a
solved model and a `SimSettings`, and it runs a panel of industries forward
under the converged policy, recording the full set of industry- and
aggregate-level objects period by period.

    sim = run_simulation(model, SimSettings(n_industries = 5_000,
                                            n_periods = 400, burnin = 100))
    sim_report(sim)

# What it records

**Per industry, per period** (matrices, `periods × industries`):

| symbol | field | definition |
|:---|:---|:---|
| `ℓ_t(j)` | `lerner` | `∑ᵢ ℓ_{i,t}(j)`, the industry Lerner index |
| `𝓜_t(j)` | `markup` | `1/(1-ℓ_t(j))`, the cost-weighted markup |
| `A_max` | `a_max` | `maxᵢ a_{i,t}(j)`, the industry frontier |
| `𝒜_t(j)` | `a_realised` | `1/(∑ᵢ s_{i,t}(j)/a_{i,t}(j))`, realised productivity |
| `Δ_t(j)` | `delta` | `A_max/𝒜`, the productive-efficiency wedge |
| `ã_t(j)` | `a_tilde` | the Cournot index (harmonic mean over active firms) |
| `ñ_t(j)` | `n_active` | active firms |
| `p_t(j)` | `price` | relative price |
| `HHI` | `hhi` | `∑ᵢ s²ᵢ = μ ℓ` |

**Aggregate, per period** (vectors):

`g_w`, `L_r`, `r`, `scriptL` (`𝓛`), `markup` (`𝓜 = 1/(1-𝓛)`), `yhat`,
`A_star` (`A* = (∫A_max^{μ-1})^{1/(μ-1)}`),
`A_within` (`(∫𝒜^{μ-1})^{1/(μ-1)}`),
`Lambda_w`, `Lambda_x` — the two wedges of the efficiency decomposition —
`n_active_mean`, `n_share` (one column per `ñ`), and `wage_check`.

`Λ^w` is not an extra assumption: substituting `ω*(j) = A_max^{μ-1}/(A*)^{μ-1}`
and `Δ = A_max/𝒜` into its definition collapses it to
`(∫𝒜^{μ-1})^{1/(μ-1)}/A*`, so it is `A_within/A_star` exactly.

# Everything is in wage units

The state is `a = A/w`, and the panel is renormalised each period so that
`∫p^{1-μ}dj = 1`, which is the definition of the price index. Hence `w = 1`
throughout and every productivity index above is expressed relative to the
wage. `wage_check` recomputes `[∫(𝒜/𝓜)^{μ-1}dj]^{1/(μ-1)}`, which
Proposition (equilibrium real wage) says must equal `w = 1`; its deviation
from one is a direct test of the static block against the aggregation.

# The ñ transition matrix

`sim.P[a,b]` estimates `P(ñ_{t+1} = b | ñ_t = a)` by counting transitions
over consecutive post-burn-in periods, and `sim.stationary` is its
invariant distribution.

Two cautions. `ñ` is *not* Markov on its own — the state is the whole
productivity profile — so `P` describes a lumped process and its rows are
averages over the profiles that happen to sit at each `ñ`. And because `P`
is estimated from transitions at stationarity, its invariant distribution
is close to the empirical distribution of `ñ` almost by construction; the
gap between `sim.stationary` and `mean(sim.agg.n_share)` therefore measures
how far the panel is from stationarity rather than validating `P`.
"""
module Simulation

using Random, Printf, Statistics, LinearAlgebra, Dates, TOML

using ..SymStateArrays
using ..StateGrids
using ..DSICModel
using ..StaticMarket
using ..ResearchPolicyFOC
using ..GeneralEquilibrium

const GE = GeneralEquilibrium

export SimSettings, IndustryPaths, AggregatePaths, SimulationOutput,
       run_simulation, industry_record, sim_report,
       transition_matrix, stationary_distribution,
       artifact_stem, write_run_config

# =====================================================================
#  1. Settings
# =====================================================================

"""
    SimSettings(; kwargs...)

How to run the panel. Independent of the model's own `Settings`, so the
same solved model can be simulated at several sample sizes without
rebuilding it.

  * `n_industries` — industries in the panel
  * `n_periods`, `burnin` — total and discarded periods
  * `thin` — record every `thin`-th post-burn-in period. The `ñ` transition
    matrix always uses *consecutive* periods regardless of `thin`, since a
    thinned transition is a different object.
  * `seed` — for the innovation draws
  * `store_industry` — keep the industry-level matrices. At
    `n_industries = 5000` and 300 recorded periods they are about 100 MB;
    set `false` for aggregates only.
"""
Base.@kwdef struct SimSettings
    n_industries::Int   = 5_000
    n_periods::Int      = 400
    burnin::Int         = 100
    thin::Int           = 1
    seed::Int           = 20260801
    store_industry::Bool = true

    function SimSettings(n_industries, n_periods, burnin, thin, seed, store_industry)
        n_industries >= 1 || throw(ArgumentError("n_industries must be ≥ 1"))
        n_periods >= 2    || throw(ArgumentError("n_periods must be ≥ 2"))
        burnin >= 0       || throw(ArgumentError("burnin must be ≥ 0"))
        burnin < n_periods - 1 || throw(ArgumentError(
            "burnin must leave at least two periods, for the transition matrix"))
        thin >= 1         || throw(ArgumentError("thin must be ≥ 1"))
        return new(n_industries, n_periods, burnin, thin, seed, store_industry)
    end
end

Base.show(io::IO, s::SimSettings) = @printf(io,
    "SimSettings(%d industries, %d periods, %d burn-in, thin %d)",
    s.n_industries, s.n_periods, s.burnin, s.thin)

# =====================================================================
#  2. One industry, fully
# =====================================================================

"""
    industry_record(profile, μ, Val(N)) -> NamedTuple

Everything the static block delivers for one industry, from a single
participation scan:

`(p, a_tilde, markup_m, n_active, lerner, hhi, a_max, a_realised, delta,
  share)`

where `markup_m` is the Cournot markup `m = μñ/(μñ-1)` on the *price*, and
the industry cost-weighted markup `𝓜 = 1/(1-ℓ)` is reported separately —
the two are different objects and the draft is careful to distinguish them.

Realised productivity `𝒜 = 1/(∑ᵢ sᵢ/aᵢ)` is the share-weighted harmonic
mean, so it counts only firms that actually produce; inactive firms carry
`sᵢ = 0` and drop out. It always weakly exceeds the Cournot index `ã`,
with equality when all active firms are equally productive, and always
falls weakly short of the frontier — so `Δ = A_max/𝒜 ≥ 1`.
"""
@inline function industry_record(profile::NTuple{N,Float64}, μ::Float64,
                                 ::Val{N}) where {N}
    meta = market_share_with_meta(profile, μ)
    p, ã, m, ñ = meta.p, meta.a_tilde, meta.m, meta.n_active

    lerner = 0.0
    hhi    = 0.0
    inv_A  = 0.0                       # ∑ᵢ sᵢ/aᵢ
    a_max  = 0.0
    own_s  = 0.0

    @inbounds for i in 1:N
        a_i = profile[i]
        a_i > a_max && (a_max = a_i)
        s_i = μ * (1.0 - ã / (m * a_i))
        s_i > 0.0 || continue
        lerner += (p - 1.0 / a_i) / p * s_i
        hhi    += s_i * s_i
        inv_A  += s_i / a_i
        i == 1 && (own_s = s_i)
    end

    a_realised = 1.0 / inv_A
    return (p = p, a_tilde = ã, markup_m = m, n_active = ñ,
            lerner = lerner, hhi = hhi, a_max = a_max,
            a_realised = a_realised, delta = a_max / a_realised,
            share = own_s, output_share = p^(1.0 - μ),
            markup_M = 1.0 / (1.0 - lerner))
end

# =====================================================================
#  3. Containers
# =====================================================================

"""
    IndustryPaths

Industry-level quantities, each a `periods × industries` matrix. Row `t` is
the cross-section in recorded period `t`; column `j` is one industry's
history. `vec(x)` pools everything, `mean(x, dims = 2)` gives the path of
the cross-sectional mean.
"""
struct IndustryPaths
    lerner::Matrix{Float64}      # ℓ_t(j)
    markup::Matrix{Float64}      # 𝓜_t(j) = 1/(1-ℓ)
    a_max::Matrix{Float64}       # A_max,t(j)
    a_realised::Matrix{Float64}  # 𝒜_t(j)
    delta::Matrix{Float64}       # Δ_t(j) = A_max/𝒜
    a_tilde::Matrix{Float64}     # ã_t(j)
    price::Matrix{Float64}       # p_t(j)
    hhi::Matrix{Float64}         # ∑ᵢsᵢ² = μℓ
    n_active::Matrix{Int}        # ñ_t(j)
end

IndustryPaths(P::Int, S::Int) =
    IndustryPaths(Matrix{Float64}(undef, P, S), Matrix{Float64}(undef, P, S),
                  Matrix{Float64}(undef, P, S), Matrix{Float64}(undef, P, S),
                  Matrix{Float64}(undef, P, S), Matrix{Float64}(undef, P, S),
                  Matrix{Float64}(undef, P, S), Matrix{Float64}(undef, P, S),
                  Matrix{Int}(undef, P, S))

IndustryPaths() = IndustryPaths(0, 0)

Base.size(x::IndustryPaths) = size(x.lerner)
Base.show(io::IO, x::IndustryPaths) =
    print(io, "IndustryPaths(", size(x, 1), " periods × ", size(x, 2),
              " industries)")
Base.size(x::IndustryPaths, d::Int) = size(x.lerner, d)

"""
    AggregatePaths

Aggregate quantities, one entry per recorded period.

`Lambda_w` and `Lambda_x` are the two terms of the efficiency
decomposition; their product is the economy's allocative efficiency.
`wage_check` should equal one — see the module docstring.
"""
struct AggregatePaths
    period::Vector{Int}
    g_w::Vector{Float64}
    L_r::Vector{Float64}
    r::Vector{Float64}           # r = (1 + g)^σ / β - 1
    scriptL::Vector{Float64}     # 𝓛
    markup::Vector{Float64}      # 𝓜 = 1/(1-𝓛)
    yhat::Vector{Float64}        # ŷ = (1-L^r)/(1-𝓛)
    A_star::Vector{Float64}      # A* = (∫A_max^{μ-1})^{1/(μ-1)}
    A_within::Vector{Float64}    # (∫𝒜^{μ-1})^{1/(μ-1)} = Λ^w·A*
    Lambda_w::Vector{Float64}
    Lambda_x::Vector{Float64}
    n_active_mean::Vector{Float64}
    n_share::Matrix{Float64}     # periods × n
    wage_check::Vector{Float64}
    outside_frac::Vector{Float64}
end

function AggregatePaths(P::Int, n::Int)
    v() = Vector{Float64}(undef, P)
    return AggregatePaths(Vector{Int}(undef, P),
                          v(), v(), v(), v(), v(), v(), # g_w … r
                          v(), v(), v(), v(), v(),      # A_star … n_active_mean
                          Matrix{Float64}(undef, P, n), # n_share
                          v(), v())                     # wage_check, outside_frac
end

Base.length(a::AggregatePaths) = length(a.period)

"""
    SimulationOutput

Everything one simulation produced. `agg` holds the aggregate paths,
`industry` the cross-sections, `P` the `ñ` transition matrix with
`stationary` its invariant distribution, and `n_counts` the raw transition
counts behind `P`.

`params`, `model_settings` and `aggs` record the model the run came from,
so the output is self-describing: [`write_run_config`](@ref) can reproduce
the whole configuration without the model being in scope.
"""
struct SimulationOutput
    settings::SimSettings
    params::Params
    model_settings::Settings
    aggs::Aggregates
    agg::AggregatePaths
    industry::IndustryPaths
    P::Matrix{Float64}
    stationary::Vector{Float64}
    n_counts::Matrix{Int}
end

Base.show(io::IO, o::SimulationOutput) = @printf(io,
    "SimulationOutput(n=%d, %d periods × %d industries, g_w=%.5f, 𝓛=%.4f)",
    o.params.n, length(o.agg), o.settings.n_industries,
    mean(o.agg.g_w), mean(o.agg.scriptL))

# =====================================================================
#  4. The transition matrix
# =====================================================================

"""
    transition_matrix(counts) -> Matrix{Float64}

Row-normalise a matrix of transition counts. A row with no observations —
an `ñ` never visited — is returned as a unit row on itself, which keeps the
matrix stochastic and leaves that state absorbing but unreachable, so it
carries zero stationary mass.
"""
function transition_matrix(counts::AbstractMatrix{Int})
    n = size(counts, 1)
    P = zeros(Float64, n, n)
    for a in 1:n
        tot = sum(@view counts[a, :])
        if tot == 0
            P[a, a] = 1.0
        else
            @inbounds for b in 1:n
                P[a, b] = counts[a, b] / tot
            end
        end
    end
    return P
end

"""
    stationary_distribution(P; tol, maxit) -> Vector{Float64}

The invariant distribution of a row-stochastic `P`, by power iteration from
uniform. Returns the fixed point of `π' = πP` normalised to sum to one.

Power iteration rather than an eigensolver because `P` here is small
(`n × n` with `n ≤ 10`), because the result is guaranteed non-negative
without post-hoc sign fixing, and because a chain with an unvisited
absorbing row would give an eigensolver a second unit eigenvalue.
"""
function stationary_distribution(P::AbstractMatrix{Float64};
                                 tol = 1e-14, maxit = 100_000)
    n = size(P, 1)
    π = fill(1.0 / n, n)
    for _ in 1:maxit
        π_new = vec(π' * P)
        s = sum(π_new)
        s > 0 && (π_new ./= s)
        maximum(abs, π_new .- π) < tol && (π = π_new; break)
        π = π_new
    end
    return π
end

# =====================================================================
#  5. The simulation
# =====================================================================

"""
    run_simulation(model::DSIC, set::SimSettings) -> SimulationOutput

Run the panel forward under `model.sol.policy` and record everything.

The policy is read by interpolation, and extrapolated where a firm has left
the grid; `agg.outside_frac` reports how often that happened. The panel is
renormalised each period so `∫p^{1-μ} = 1`, and the factor that requires is
that period's `1 + g_w`.

The industry loop is threaded and every reduction runs sequentially over
per-industry buffers, so results do not depend on `Threads.nthreads()`.

`model.sol` is not modified.
"""
function run_simulation(model::DSIC, set::SimSettings)
    N = state_length(model.params)
    return _run(model, set, Val(N))
end

run_simulation(model::DSIC; kwargs...) =
    run_simulation(model, SimSettings(; kwargs...))

function _run(model::DSIC, set::SimSettings, ::Val{N}) where {N}
    par = model.params
    μ, γ = par.μ, par.γ
    haz  = Hazard(par.η̄, par.θ, par.ε)
    π̃    = Interpolant(model.sol.policy, model.grid)
    lo, hi = extrema(xaxis(model.grid))

    S = set.n_industries
    T = set.n_periods
    n = par.n

    rec  = [t for t in 1:T if t > set.burnin && (t - set.burnin) % set.thin == 0]
    Pn   = length(rec)
    slot = zeros(Int, T)
    for (pi, t) in enumerate(rec)
        slot[t] = pi
    end

    agg  = AggregatePaths(Pn, n)
    ind  = set.store_industry ? IndustryPaths(Pn, S) : IndustryPaths()

    # --- draws, firm-major ------------------------------------------
    rng = MersenneTwister(set.seed)
    u   = rand(rng, N, S, T)
    A   = lo .+ (hi - lo) .* rand(rng, N, S)

    # --- per-industry scratch ---------------------------------------
    b_post   = Vector{Float64}(undef, S)   # p^{1-μ} after innovation
    b_share  = zeros(S)                    # p^{1-μ} before
    b_sl     = zeros(S)                    # share × ℓ
    b_amax   = zeros(S)                    # A_max^{μ-1}
    b_areal  = zeros(S)                    # 𝒜^{μ-1}
    b_wage   = zeros(S)                    # (𝒜/𝓜)^{μ-1}
    b_res    = zeros(S)                    # ∑ᵢ l
    b_out    = zeros(Int, S)
    nact     = zeros(Int, S)               # ñ this period
    nprev    = zeros(Int, S)               # ñ last period
    counts   = zeros(Int, n, n)

    A ./= (sum(GE.level_shares!(b_post, A, μ, S, Val(N))) / S)^(1 / (μ - 1))

    have_prev = false

    for t in 1:T
        post   = t > set.burnin
        record = slot[t] != 0
        pi     = slot[t]

        Threads.@threads for s in 1:S
            r = industry_record(GE.firm_view(A, s, 1, Val(N)), μ, Val(N))
            @inbounds nact[s] = r.n_active

            if post
                @inbounds begin
                    b_share[s] = r.output_share
                    b_sl[s]    = r.output_share * r.lerner
                    b_amax[s]  = r.a_max^(μ - 1)
                    b_areal[s] = r.a_realised^(μ - 1)
                    b_wage[s]  = (r.a_realised / r.markup_M)^(μ - 1)
                    b_res[s]   = 0.0
                    b_out[s]   = 0
                end
            end

            if record && set.store_industry
                @inbounds begin
                    ind.lerner[pi, s]     = r.lerner
                    ind.markup[pi, s]     = r.markup_M
                    ind.a_max[pi, s]      = r.a_max
                    ind.a_realised[pi, s] = r.a_realised
                    ind.delta[pi, s]      = r.delta
                    ind.a_tilde[pi, s]    = r.a_tilde
                    ind.price[pi, s]      = r.p
                    ind.hhi[pi, s]        = r.hhi
                    ind.n_active[pi, s]   = r.n_active
                end
            end

            @inbounds for i in 1:N
                v = GE.firm_view(A, s, i, Val(N))
                l = max(π̃(v), 0.0)
                η = innovation_prob(l, v[1], haz)
                if post
                    b_res[s] += l
                    (lo <= v[1] <= hi) || (b_out[s] += 1)
                end
                u[i, s, t] < η && (A[i, s] *= γ)
            end

            @inbounds b_post[s] =
                GE.industry_stats(GE.firm_view(A, s, 1, Val(N)), μ, Val(N)).share
        end

        # --- sequential reductions -----------------------------------
        factor = (sum(b_post) / S)^(1 / (μ - 1))
        A    ./= factor

        # --- ñ transitions, always on consecutive periods -------------
        if post && have_prev
            @inbounds for s in 1:S
                counts[nprev[s], nact[s]] += 1
            end
        end
        post && (copyto!(nprev, nact); have_prev = true)

        if record
            den   = sum(b_share) / S
            𝓛     = (sum(b_sl) / S) / den
            L_r   = sum(b_res) / S
            A_st  = (sum(b_amax) / S)^(1 / (μ - 1))
            A_wi  = (sum(b_areal) / S)^(1 / (μ - 1))
            wchk  = (sum(b_wage) / S)^(1 / (μ - 1))

            @inbounds begin
                agg.period[pi]        = t
                agg.g_w[pi]           = factor - 1.0
                agg.L_r[pi]           = L_r
                agg.scriptL[pi]       = 𝓛
                agg.markup[pi]        = 1.0 / (1.0 - 𝓛)
                agg.yhat[pi]          = (1 - L_r) / (1 - 𝓛)
                agg.A_star[pi]        = A_st
                agg.A_within[pi]      = A_wi
                agg.Lambda_w[pi]      = A_wi / A_st
                agg.Lambda_x[pi]      = _lambda_x(b_areal, b_share, μ, S, A_wi)
                agg.n_active_mean[pi] = sum(nact) / S
                agg.wage_check[pi]    = wchk
                agg.outside_frac[pi]  = sum(b_out) / (S * N)
                agg.r[pi]             = (1.0 + agg.g_w[pi])^par.σ / par.β - 1.0
                for k in 1:n
                    agg.n_share[pi, k] = count(==(k), nact) / S
                end
            end
        end
    end

    Pmat = transition_matrix(counts)
    return SimulationOutput(set, par, model.settings, model.sol.aggs,
                            agg, ind, Pmat,
                            stationary_distribution(Pmat), counts)
end

"""
    _lambda_x(areal, share, μ, S, A_within) -> Float64

`Λ^x = (∫ω 𝓜^{-(μ-1)})^{μ/(μ-1)} / (∫ω 𝓜^{-μ})` with `ω ∝ 𝒜^{μ-1}`.

Rather than carry a third buffer, note that `p = 𝓜/𝒜` in wage units, so
`ω(j)𝓜(j)^{-(μ-1)} ∝ 𝒜^{μ-1}(𝒜/𝓜)^{... }` — the two moments needed are
recoverable from `𝒜^{μ-1}` and the output share `p^{1-μ} = (𝒜/𝓜)^{μ-1}`.
Concretely `𝓜^{-(μ-1)} = share/areal` and `𝓜^{-μ} = 𝓜^{-(μ-1)}·𝒜/𝓜`,
so both integrals are sums over the two buffers already filled.
"""
function _lambda_x(areal::Vector{Float64}, share::Vector{Float64},
                   μ::Float64, S::Int, A_within::Float64)
    # ω(j) = 𝒜^{μ-1} / ∫𝒜^{μ-1};  𝓜^{-(μ-1)} = share/areal
    num = 0.0     # ∫ ω 𝓜^{-(μ-1)}
    den = 0.0     # ∫ ω 𝓜^{-μ}
    @inbounds for s in eachindex(areal, share)
        w  = areal[s]
        mi = w > 0 ? share[s] / w : 0.0          # 𝓜^{-(μ-1)}
        num += w * mi
        # 𝓜^{-μ} = (𝓜^{-(μ-1)})^{μ/(μ-1)}
        den += w * mi^(μ / (μ - 1))
    end
    tot = A_within^(μ - 1) * S
    num /= tot
    den /= tot
    return den > 0 ? num^(μ / (μ - 1)) / den : NaN
end

# =====================================================================
#  6. Saving: one stem, several files
# =====================================================================

"""
    artifact_stem(save, tag) -> String

Build a timestamped path stem from a `save` argument of the form
`"directory/prefix"`, creating the directory if needed:

    artifact_stem("output/baseline", "industry")
      → "output/baseline_industry_20260804-143012"

Callers append their own extensions to the returned stem, so a figure and
its configuration file share a name and differ only in extension. Compute
the stem **once** per artifact and reuse it — calling this twice would
stamp two different seconds and break the pairing.

The timestamp is local time to the second, which is enough to keep
successive runs apart without making the names unreadable.
"""
function artifact_stem(save::AbstractString, tag::AbstractString)
    dir = dirname(save)
    pre = basename(save)
    isempty(pre) && throw(ArgumentError(
        "`save` must end in a filename prefix, e.g. \"output/baseline\" " *
        "(got $(repr(save)))"))
    isempty(dir) && (dir = ".")
    mkpath(dir)
    stamp = Dates.format(Dates.now(), "yyyymmdd-HHMMSS")
    return joinpath(dir, string(pre, "_", tag, "_", stamp))
end

# TOML has no Symbol type, so write symbols as strings on the way out.
# NaN is representable (TOML 1.0 allows `nan`), so ymin/ymax pass through.
_tv(v::Symbol) = String(v)
_tv(v) = v
_tdict(x) = Dict{String,Any}(k => _tv(v) for (k, v) in to_dict(x))

"""
    write_run_config(path, o::SimulationOutput; extra = Dict())

Write the configuration behind a simulation to `path` as TOML: the economic
parameters, the model's numerical settings, the simulation settings, and
the equilibrium the model had reached.

Every saved figure and report is paired with one of these under the same
name, so a plot on disk can always be traced back to the run that made it.
Pass `extra` to record anything else — a git commit, a note about the
experiment.

The file is readable by `load_config`, so a calibration can be recovered
from any saved artifact.
"""
function write_run_config(path::AbstractString, o::SimulationOutput;
                          extra::AbstractDict = Dict{String,Any}())
    doc = Dict{String,Any}(
        "schema"    => SCHEMA_VERSION,
        "generated" => string(Dates.now()),
        "params"    => _tdict(o.params),
        "settings"  => _tdict(o.model_settings),
        "sim"       => _tdict(o.settings),
        "equilibrium" => Dict{String,Any}(
            "g_w"          => o.aggs.g_w,
            "g_y"          => o.aggs.g_y,
            "yhat"         => o.aggs.ŷ,
            "r"            => (1 + o.aggs.g_y)^o.params.σ / o.params.β - 1,
            "g_w_sim"      => mean(o.agg.g_w),
            "L_r_sim"      => mean(o.agg.L_r),
            "scriptL_sim"  => mean(o.agg.scriptL),
            "markup_sim"   => mean(o.agg.markup),
            "Lambda_w_sim" => mean(o.agg.Lambda_w),
            "Lambda_x_sim" => mean(o.agg.Lambda_x),
            "wage_check"   => mean(o.agg.wage_check)))
    for (k, v) in extra
        doc[String(k)] = v
    end
    open(path, "w") do io
        TOML.print(io, doc; sorted = true)
    end
    return path
end

# =====================================================================
#  7. Reporting
# =====================================================================

qt(x) = (quantile(x, 0.10), median(x), quantile(x, 0.90))

function _line(io::IO, name, x; d = 4)
    q10, q50, q90 = qt(x)
    @printf(io, "%-22s mean %9.*f  sd %8.*f  [p10 %8.*f  p50 %8.*f  p90 %8.*f]\n",
            name, d, mean(x), d, std(x), d, q10, d, q50, d, q90)
end

"""
    sim_report(o::SimulationOutput; save = nothing, extra = Dict())

Print the aggregate paths, the industry cross-section, and the `ñ`
transition matrix with its invariant distribution.

`wage_check` should print as `1.000000`: it recomputes the equilibrium real
wage from the simulated cross-section by a route the simulation never uses,
so a deviation means the static block and the aggregation disagree.

With `save = "output/baseline"` the same text is written to
`output/baseline_report_<timestamp>.simres`, alongside a `.toml` of the run
configuration under the identical name. The text is rendered once into a
buffer and then both printed and written, so the file and the terminal can
never disagree.

Returns the path stem when saving, `nothing` otherwise.
"""
function sim_report(o::SimulationOutput; save = nothing,
                    extra::AbstractDict = Dict{String,Any}())
    buf = IOBuffer()
    _sim_report(buf, o)
    txt = String(take!(buf))
    print(txt)
    save === nothing && return nothing

    stem = artifact_stem(save, "report")
    write(stem * ".simres", txt)
    write_run_config(stem * ".toml", o; extra = extra)
    @info "saved report" text = stem * ".simres" config = stem * ".toml"
    return stem
end

function _sim_report(io::IO, o::SimulationOutput)
    a, ind, par = o.agg, o.industry, o.params
    n = par.n

    println(io, "\n", "═"^92)
    println(io, "SIMULATION   ", o.settings)
    println(io, "═"^92)
    _line(io, "g_w", a.g_w; d = 6)
    _line(io, "r",   a.r;   d = 6)
    _line(io, "L^r", a.L_r; d = 6)
    _line(io, "𝓛   aggregate Lerner", a.scriptL)
    _line(io, "𝓜 = 1/(1-𝓛)", a.markup)
    _line(io, "ŷ", a.yhat)
    _line(io, "A*  frontier index", a.A_star)
    _line(io, "(∫𝒜^{μ-1})^{1/(μ-1)}", a.A_within)
    _line(io, "Λ^w within", a.Lambda_w)
    _line(io, "Λ^x across", a.Lambda_x)
    _line(io, "Λ = Λ^w Λ^x", a.Lambda_w .* a.Lambda_x)
    _line(io, "mean ñ", a.n_active_mean)
    @printf(io, "%-22s %9.6f   (must be 1; deviation = static/aggregate gap)\n",
            "wage check", mean(a.wage_check))
    @printf(io, "%-22s %9.3f%%  ← policy extrapolated past the grid\n",
            "states outside grid", 100 * mean(a.outside_frac))

    if size(ind, 1) > 0
        println(io, "\n", "─"^92)
        println(io, "INDUSTRY CROSS-SECTION   (", size(ind, 1), " periods × ",
                size(ind, 2), " industries)")
        println(io, "─"^92)
        _line(io, "ℓ(j)  Lerner", vec(ind.lerner))
        _line(io, "𝓜(j) = 1/(1-ℓ)", vec(ind.markup))
        _line(io, "HHI(j) = μℓ(j)", vec(ind.hhi))
        _line(io, "A_max(j)", vec(ind.a_max))
        _line(io, "𝒜(j)  realised", vec(ind.a_realised))
        _line(io, "ã(j)  Cournot", vec(ind.a_tilde))
        _line(io, "Δ(j) = A_max/𝒜", vec(ind.delta))
        _line(io, "p(j)", vec(ind.price))
    end

    println(io, "\n", "─"^92)
    println(io, "ACTIVE-FIRM TRANSITIONS   P(ñ → ñ′)")
    println(io, "─"^92)
    print(io, "      ")
    for b in 1:n
        @printf(io, "%10s", "→ $b")
    end
    println(io, "      (count)")
    for aa in 1:n
        @printf(io, "ñ = %-2d", aa)
        for b in 1:n
            @printf(io, "%10.4f", o.P[aa, b])
        end
        @printf(io, "   %10d\n", sum(@view o.n_counts[aa, :]))
    end
    println(io)
    print(io, "stationary  ")
    for k in 1:n
        @printf(io, "%10.4f", o.stationary[k])
    end
    println(io)
    print(io, "empirical   ")
    emp = [mean(@view a.n_share[:, k]) for k in 1:n]
    for k in 1:n
        @printf(io, "%10.4f", emp[k])
    end
    @printf(io, "      max gap %.2e\n", maximum(abs, o.stationary .- emp))

    # persistence: how often an industry keeps the same ñ
    diag_mass = sum(o.stationary[k] * o.P[k, k] for k in 1:n)
    @printf(io, "\n%-22s %9.4f   (probability ñ is unchanged next period)\n",
            "persistence", diag_mass)
    println(io, "═"^92)
    return nothing
end

end # module