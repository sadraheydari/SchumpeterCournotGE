"""
    GeneralEquilibrium

The outer loop. Guess the aggregates, solve the symmetric policy
equilibrium, simulate the economy, read the aggregates back out, damp
toward them, repeat.

    Aggregates ──▶ solve_symmetric_policy! ──▶ simulate_economy! ──▶ Aggregates'
         ▲                                                              │
         └──────────────── damp(new, old, λ_agg) ◀──────────────────────┘

# How g_w is obtained, and why not by iteration

`∫p(j)^{1-μ}dj = 1` is not an equilibrium residual — it is the *definition*
of the price index, and hence of the level of `w`. The simulation therefore
**imposes it every period** and reads the growth rate off the
renormalisation, rather than carrying `g_w` in the transition and checking
it afterwards.

Scaling every productivity by `c` leaves the participation set and the
markup untouched (both sides of the entry condition scale together) and
divides `p` by `c`. So after innovation, with profiles `ã = γ^s·a`,

    1 + g_w,t = [ ∫ p(ã)^{1-μ} dj ]^{1/(μ-1)} ,        a_{t+1} = ã / (1+g_w,t)

leaves `∫p^{1-μ} = 1` intact. This is the draft's wage equation, applied one
period at a time, and it handles entry and exit automatically since `p` is
recomputed from the new profile.

The alternative — fixing `g_w` in the transition and comparing the implied
growth — looks reasonable and is empty: at a stationary distribution the
implied growth equals the assumed growth *identically*, for any `g_w` that
admits a stationary distribution. Here the guessed `g_w` enters only through
the **policy**, so the map from guess to outcome has real content.

# Layout and parallelism

The panel is stored **firm-major**, `A[i, s]` for firm `i` of industry `s`,
so an industry's whole profile occupies `8n` contiguous bytes — one cache
line for `n ≤ 8`. The obvious `S × n` layout puts an industry's
productivities `8S` bytes apart (16 KB at `S = 2000`), so every `firm_view`
would touch `n` separate cache lines and be called `n` times per industry.
The shock array `u[i, s, t]` follows the same convention.

Within a period, industries are independent: industry `s` reads and writes
only column `s` and its own shocks. The industry loop is therefore threaded.
The period loop is not, and cannot be — `A` at `t+1` depends on `t`.

**Reductions stay deterministic.** Each thread writes per-industry
contributions into preallocated buffers at distinct indices; the sums are
then taken sequentially over those buffers. Accumulating into per-thread
partials instead would make the answer depend on `Threads.nthreads()`
through the order of floating-point addition — which would defeat the
common random numbers below, whose whole purpose is a deterministic map.

# Common random numbers

The shocks are drawn once, in [`SimulationDraws`](@ref), and reused at every
outer iteration. Fresh draws each time would make the whole map noisy: the
aggregates would rattle around at the Monte Carlo noise floor and never
settle, and any acceleration scheme would read the noise as curvature and
diverge. With the draws fixed the map is deterministic and smooth in the
aggregates, which is what a fixed-point iteration assumes.

# What the simulation tracks

  * `g_w` — from the renormalisation above, averaged over the post-burn-in
    periods
  * `L_r` — total research labour, since every industry has `n` firms and
    industries have unit mass
  * `ℒ` — the output-share-weighted Lerner index, `∫s(j)ℓ(j)dj / ∫s(j)dj`
  * `ŷ` — `(1-L_r)/(1-ℒ)`, the demand shifter that follows from them
"""
module GeneralEquilibrium

using Random, Printf

using ..SymStateArrays
using ..StateGrids
using ..DSICModel
using ..StaticMarket
using ..ResearchPolicyFOC
using ..ValueIteration
using ..SymPolicyEquilibrium

export SimulationDraws, SimulationResult, simulate_economy!,
       solve_equilibrium!, feasible_growth_floor

# =====================================================================
#  1. Common random numbers
# =====================================================================

"""
    SimulationDraws(model::DSIC)

Every random number the simulation will ever need, drawn once from
`settings.seed`.

  * `u[i, s, t]` — the uniform deciding whether firm `i` of industry `s`
    innovates in period `t`
  * `a0[i, s]` — the starting productivity profile

Both are **firm-major**: an industry's entries are contiguous. See the
module docstring.

Reused unchanged across outer iterations, which is what makes the map from
guessed aggregates to implied aggregates deterministic. Build one per model
and hold on to it; rebuilding mid-solve reintroduces the noise.

Size is `n × n_sims × n_periods` doubles — at 5 firms, 1000 industries and
500 periods, 20 MB.
"""
struct SimulationDraws
    u::Array{Float64,3}
    a0::Matrix{Float64}
end

function SimulationDraws(model::DSIC)
    set = model.settings
    n   = state_length(model.params)
    rng = MersenneTwister(set.seed)
    u   = rand(rng, n, set.n_sims, set.n_periods)
    # start spread over the grid; the level is fixed below and burn-in
    # washes out whatever is left of the initial condition
    lo, hi = extrema(xaxis(model.grid))
    a0 = lo .+ (hi - lo) .* rand(rng, n, set.n_sims)
    return SimulationDraws(u, a0)
end

# =====================================================================
#  2. One industry
# =====================================================================

"""
    firm_view(A, s, i, Val(N)) -> NTuple{N,Float64}

Industry `s` as its firm `i` sees it: that firm's productivity first, the
others after — the layout `StateArray` and the policy interpolant expect.

`A` is firm-major, `A[i, s]`, so this reads one contiguous run of memory.
Allocation-free.
"""
@inline function firm_view(A::Matrix{Float64}, s::Int, i::Int,
                           ::Val{N}) where {N}
    return ntuple(Val(N)) do k
        if k == 1
            @inbounds A[i, s]
        else
            j = k - 1
            @inbounds j < i ? A[j, s] : A[j + 1, s]
        end
    end
end

"""
    industry_stats(profile, μ, Val(N)) -> (share, lerner_sum, n_active)

The industry's share of aggregate output `s(j) = p^{1-μ}`, the sum of its
firms' Lerner coefficients `∑ℓᵢ`, and how many firms are active.

One participation scan for the whole industry, then `n` cheap per-firm
evaluations — rather than calling `lerner_coefficient` once per firm and
redoing the scan each time.
"""
@inline function industry_stats(profile::NTuple{N,Float64}, μ::Float64,
                                ::Val{N}) where {N}
    meta = market_share_with_meta(profile, μ)
    p, ã, m = meta.p, meta.a_tilde, meta.m

    lerner_sum = 0.0
    @inbounds for i in 1:N
        a_i = profile[i]
        s_i = μ * (1.0 - ã / (m * a_i))
        s_i > 0.0 || continue
        lerner_sum += (p - 1.0 / a_i) / p * s_i      # ℓ_i, as StaticMarket has it
    end

    return (share = industry_output_share(p, μ),
            lerner_sum = lerner_sum,
            n_active = meta.n_active)
end

# =====================================================================
#  3. Level normalisation
# =====================================================================

"""
    level_shares!(buf, A, μ, S, Val(N))

Write each industry's `p^{1-μ}` into `buf`, in parallel. Writes only, one
index per industry, so no reduction happens here and the result does not
depend on the thread count.
"""
function level_shares!(buf::Vector{Float64}, A::Matrix{Float64}, μ::Float64,
                       S::Int, ::Val{N}) where {N}
    Threads.@threads for s in 1:S
        @inbounds buf[s] = industry_stats(firm_view(A, s, 1, Val(N)), μ, Val(N)).share
    end
    return buf
end

"""
    _level_factor(A, μ, S, Val(N); buf) -> Float64

`[∫p^{1-μ}dj]^{1/(μ-1)}`, the factor every productivity must be divided by
for the price index to normalise to one. Dividing by it is exactly the
draft's wage equation, and the factor itself is `1 + g_w` for the period.

The sum runs sequentially over `buf`, so the result is bit-identical
whatever `Threads.nthreads()` happens to be.
"""
function _level_factor(A::Matrix{Float64}, μ::Float64, S::Int, ::Val{N};
                       buf::Vector{Float64} = Vector{Float64}(undef, S)) where {N}
    level_shares!(buf, A, μ, S, Val(N))
    return (sum(buf) / S)^(1 / (μ - 1))
end

# =====================================================================
#  4. The simulation
# =====================================================================

"""
    SimulationResult

What one pass of the economy implies. `g_w`, `L_r` and `ℒ` are averages over
the post-burn-in periods; `ŷ` follows from the last two.

`outside_frac` is the share of simulated firm states that fell outside the
grid and had their policy extrapolated. A few per cent is unremarkable; a
large share means the grid is too narrow and the policy is being invented
rather than interpolated.
"""
struct SimulationResult
    g_w::Float64
    L_r::Float64
    ℒ::Float64
    ŷ::Float64
    n_active::Float64
    a_mean::Float64
    outside_frac::Float64
end

Base.show(io::IO, r::SimulationResult) = @printf(io,
    "SimulationResult(g_w=%.6f, L_r=%.6f, ℒ=%.6f, ŷ=%.6f, ñ=%.3f, outside=%.1f%%)",
    r.g_w, r.L_r, r.ℒ, r.ŷ, r.n_active, 100 * r.outside_frac)

# Per-industry scratch, allocated once per simulation and reused each period.
struct SimBuffers
    post_share::Vector{Float64}   # p^{1-μ} after innovation — the level factor
    pre_share::Vector{Float64}    # p^{1-μ} before innovation — the ℒ weight
    lerner::Vector{Float64}       # ∑ᵢ ℓᵢ before innovation
    research::Vector{Float64}     # ∑ᵢ lᵢ
    n_active::Vector{Float64}
    a_sum::Vector{Float64}
    outside::Vector{Int}
end

SimBuffers(S::Int) = SimBuffers(Vector{Float64}(undef, S), zeros(S), zeros(S),
                                zeros(S), zeros(S), zeros(S), zeros(Int, S))

"""
    simulate_economy!(model::DSIC, draws::SimulationDraws) -> SimulationResult

Run the economy forward under the current policy and report the aggregates
it implies.

Each period, for every industry: read each firm's research off the policy
(interpolating, and extrapolating where a firm has left the grid), draw who
innovates, renormalise so that `∫p^{1-μ} = 1` still holds, and record the
growth that renormalisation required.

The industry loop is threaded; the period loop is sequential by necessity.
Statistics are accumulated only after `burnin` periods. `model.sol` is not
modified.
"""
function simulate_economy!(model::DSIC, draws::SimulationDraws)
    N = state_length(model.params)
    return _simulate(model, draws, Val(N))
end

function _simulate(model::DSIC, draws::SimulationDraws, ::Val{N}) where {N}
    par, set, sol = model.params, model.settings, model.sol
    μ, γ = par.μ, par.γ
    haz  = Hazard(par.η̄, par.θ, par.ε)
    π̃    = Interpolant(sol.policy, model.grid)      # by reference: stays current
    lo, hi = extrema(xaxis(model.grid))

    S = set.n_sims
    A = copy(draws.a0)                              # firms × industries
    b = SimBuffers(S)

    # put the starting panel on the equilibrium level, so period one is not
    # spent undoing an arbitrary initial scale
    A ./= _level_factor(A, μ, S, Val(N); buf = b.post_share)

    kept      = 0
    acc_gw    = 0.0
    acc_Lr    = 0.0
    acc_num_ℒ = 0.0
    acc_den_ℒ = 0.0
    acc_nact  = 0.0
    acc_amean = 0.0
    outside   = 0
    seen      = 0

    for t in 1:set.n_periods
        record = t > set.burnin

        Threads.@threads for s in 1:S
            # --- before anyone innovates ------------------------------
            if record
                st = industry_stats(firm_view(A, s, 1, Val(N)), μ, Val(N))
                @inbounds begin
                    b.pre_share[s] = st.share
                    b.lerner[s]    = st.lerner_sum
                    b.n_active[s]  = st.n_active
                    b.research[s]  = 0.0
                    b.a_sum[s]     = 0.0
                    b.outside[s]   = 0
                end
            end

            @inbounds for i in 1:N
                view_i = firm_view(A, s, i, Val(N))
                a_own  = view_i[1]
                l      = max(π̃(view_i), 0.0)
                η      = innovation_prob(l, a_own, haz)

                if record
                    b.research[s] += l
                    b.a_sum[s]    += a_own
                    (lo <= a_own <= hi) || (b.outside[s] += 1)
                end

                draws.u[i, s, t] < η && (A[i, s] *= γ)
            end

            # --- and after, for the renormalisation -------------------
            @inbounds b.post_share[s] =
                industry_stats(firm_view(A, s, 1, Val(N)), μ, Val(N)).share
        end

        # --- sequential reductions: independent of the thread count ---
        factor = (sum(b.post_share) / S)^(1 / (μ - 1))
        A    ./= factor

        if record
            kept      += 1
            acc_gw    += factor - 1.0               # 1 + g_w,t = factor
            acc_Lr    += sum(b.research) / S
            acc_num_ℒ += _dotsum(b.pre_share, b.lerner) / S
            acc_den_ℒ += sum(b.pre_share) / S
            acc_nact  += sum(b.n_active) / S
            acc_amean += sum(b.a_sum) / (S * N)
            outside   += sum(b.outside)
            seen      += S * N
        end
    end

    kept > 0 || throw(ArgumentError(
        "burnin ($(set.burnin)) leaves no periods to average over " *
        "(n_periods = $(set.n_periods))"))

    g_w = acc_gw / kept
    L_r = acc_Lr / kept
    ℒ   = acc_den_ℒ > 0 ? acc_num_ℒ / acc_den_ℒ : NaN
    frac_out = seen > 0 ? outside / seen : 0.0

    # Nothing in the simulation bounds research by the labour endowment.
    # Off-equilibrium it can breach it — most often because firms drifted
    # above the grid and their policy was extrapolated rather than
    # interpolated, and linear extrapolation of a rising policy has no
    # ceiling. Say so, rather than letting yhat_from complain about a
    # demand shifter.
    L_r < 1 || throw(ArgumentError(
        "simulated research labour L_r = $L_r exceeds the endowment of 1, " *
        "so ŷ = (1-L_r)/(1-ℒ) is not positive. $(round(100*frac_out, digits=1))% " *
        "of firm states fell outside the grid and had their policy " *
        "extrapolated; widen the grid, lower ŷ, or start the aggregate loop " *
        "closer to its fixed point."))
    ℒ < 1 || throw(ArgumentError(
        "simulated Lerner index ℒ = $ℒ reached 1, so ŷ = (1-L_r)/(1-ℒ) is " *
        "undefined; the industry profit share cannot exhaust output"))

    return SimulationResult(g_w, L_r, ℒ, yhat_from(L_r, ℒ),
                            acc_nact / kept, acc_amean / kept, frac_out)
end

"Sequential dot product, so the reduction order is fixed."
@inline function _dotsum(x::Vector{Float64}, y::Vector{Float64})
    s = 0.0
    @inbounds @simd for i in eachindex(x, y)
        s += x[i] * y[i]
    end
    return s
end

# =====================================================================
#  5. The outer loop
# =====================================================================

"""
    feasible_growth_floor(par::Params) -> Float64

The smallest `g_y` for which the Bellman operator is a contraction:
`Λ = β(1+g_y)^{-σ}γ < 1` rearranges to `g_y > (βγ)^{1/σ} - 1`.

A model with `γ > 1` therefore has **no bounded solution at zero growth**.
The outer loop can propose a candidate below this floor while it is still
far from the answer, so [`solve_equilibrium!`](@ref) lifts any such
candidate back above it rather than handing the value iteration a problem
with no solution.
"""
feasible_growth_floor(par::Params) = (par.β * par.γ)^(1 / par.σ) - 1

"""
    solve_equilibrium!(model, ws, draws; kwargs...) -> LoopStatus

Iterate the aggregates to a fixed point, writing the result into
`model.sol` and the outcome into `model.sol.agg`.

Each pass solves the symmetric policy equilibrium at the current
aggregates, simulates, and damps toward what the simulation implies. The
per-iteration residual is `dist(new, old)` — the sup-norm over
`(g_w, g_y, ŷ)` — recorded in `sol.history`.

Keywords:

  * `λ = model.settings.λ_agg` — damping on the aggregate update
  * `require_inner = true` — refuse to report convergence while the
    symmetric-policy loop underneath is failing
  * `on_iter(iteration, residual, result)` — after each aggregate pass
  * `on_sym`, `on_vfi` — passed down to the inner loops

`g_y` is set equal to `g_w`, the balanced-path restriction. Nothing is
reset between iterations: the value function, the rival belief and the
aggregates all warm start, which is why later passes are far cheaper than
the first.
"""
function solve_equilibrium!(model::DSIC, ws::VFIWorkspace,
                            draws::SimulationDraws;
                            λ::Real = model.settings.λ_agg,
                            require_inner::Bool = true,
                            on_iter = nothing,
                            on_sym = nothing,
                            on_vfi = nothing)
    set, sol, par = model.settings, model.sol, model.params
    floor_g = feasible_growth_floor(par)

    sol.agg.converged = false
    sol.agg.iters     = 0
    sol.agg.residual  = Inf
    empty!(sol.history)

    inner_failures = 0

    for iter in 1:set.maxiter_agg
        # --- 1. the firms' problem at the current aggregates ----------
        solve_symmetric_policy!(model, ws; on_iter = on_sym, on_vfi = on_vfi)
        sol.sym_policy.converged || (inner_failures += 1)

        # --- 2. run the economy ---------------------------------------
        res = simulate_economy!(model, draws)

        # --- 3. the residual, measured before the update --------------
        proposed = Aggregates(res.g_w, res.g_w, res.ŷ)   # g_y = g_w on a BGP
        residual = dist(proposed, sol.aggs)

        # --- 4. damped step -------------------------------------------
        updated = damp(proposed, sol.aggs, λ)

        # A candidate below the contraction floor would hand the value
        # iteration a problem with no bounded solution. Lift it rather
        # than diverge; if it keeps happening the equilibrium is outside
        # the range these parameters can support.
        if contraction_modulus(par, updated) >= 1
            lifted = floor_g + 1e-4
            @warn("the damped growth rate is below the contraction floor; " *
                  "lifting it", proposed = updated.g_y, floor = floor_g,
                  lifted_to = lifted, iteration = iter)
            updated = Aggregates(lifted, lifted, updated.ŷ)
        end

        sol.aggs = updated
        sol.L_r  = res.L_r
        sol.ℒ    = res.ℒ

        sol.agg.iters    = iter
        sol.agg.residual = residual
        push!(sol.history, residual)
        on_iter === nothing || on_iter(iter, residual, res)

        if residual < set.tol_agg
            sol.agg.converged = !(require_inner && !sol.sym_policy.converged)
            break
        end
    end

    if inner_failures > 0
        @warn("$inner_failures of $(sol.agg.iters) symmetric-policy solves did " *
              "not converge; the simulations they fed were run on unreliable " *
              "policies", tol_sym_policy = set.tol_sym_policy,
              maxiter_sym_policy = set.maxiter_sym_policy)
    end

    return sol.agg
end

"""
    solve_equilibrium!(model::DSIC; mode = :auto, kwargs...) -> LoopStatus

Convenience form that builds the workspace and the random draws. Both are
expensive to construct, so keep them if you intend to solve the same model
more than once.
"""
function solve_equilibrium!(model::DSIC; mode::Symbol = :auto, kwargs...)
    ws    = VFIWorkspace(model; mode = mode)
    draws = SimulationDraws(model)
    return solve_equilibrium!(model, ws, draws; kwargs...)
end

end # module