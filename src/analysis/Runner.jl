"""
    Runner

One call from parameters to results, for any number of firms.

    res = run_model(n = 2, η̄ = 2.0, γ = 1.06)

`run_model` builds the model, solves all three loops, runs the economy, and
collects the cross-section — not just its averages, so the distributions
can be looked at rather than summarised away.

# Two panels

  * **`res.panel`** — the economy started from a dispersed cross-section,
    the ergodic distribution of industries.
  * **`res.sym_panel`** — the same economy with every industry beginning
    with all `n` firms at *identical* productivity. Nothing distinguishes
    them at date zero; they come apart only through the innovation draws.

Since the ergodic distribution does not depend on where you start, the two
should agree on within-industry statistics. [`report`](@ref) prints both so
you can see whether they do — if not, `burnin` is too short.

Both reuse `GeneralEquilibrium`'s own `firm_view`, `industry_stats` and
`_level_factor`, so the recorded dynamics cannot drift from the ones the
solver used.
"""
module Runner

using Printf, Statistics

using ..SymStateArrays
using ..StateGrids
using ..DSICModel
using ..StaticMarket
using ..ResearchPolicyFOC
using ..ValueIteration
using ..SymPolicyEquilibrium
using ..GeneralEquilibrium
using ..ProgressBars

const GE = GeneralEquilibrium

export Panel, ModelResult, build_model, run_model, solve_model!,
       simulate_panel, symmetric_start_panel, report, binned_mean

# =====================================================================
#  1. Building
# =====================================================================

"""
    build_model(; kwargs...) -> DSIC

A model with `n` firms per industry on a grid over de-trended productivity
`a = A/w`.

`kx == ky` throughout, so the `:exact` rival lookup is available: a rival's
view of a state permutes the first component with a symmetric one, which is
only itself a grid state when the two axes are the same grid.

Economic parameters: `n, β, σ, μ, γ, θ, ε, η̄`.
Grid: `amin, amax, k, spacing, spacing_param`.
Loops: `tol_vfi/maxiter_vfi`, `tol_sym/maxiter_sym/λ_sym`,
`tol_agg/maxiter_agg/λ_agg`.
Starting aggregates: `g_w, g_y, ŷ`.
Simulation: `n_sims, n_periods, burnin, seed`.

Watch the state count: it is `k · C(k+n-2, n-1)`, which grows fast in `n`.
"""
function build_model(; n = 2, β = 0.94, σ = 2.0, μ = 2.0, γ = 1.06,
                       θ = 0.30, ε = 0.50, η̄ = 1.5,
                       amin = 0.25, amax = 8.0, k = 60,
                       spacing = :power, spacing_param = 1.4,
                       g_w = 0.01, g_y = 0.01, ŷ = 1.8,
                       tol_vfi = 1e-10, maxiter_vfi = 8_000,
                       tol_sym = 1e-7, maxiter_sym = 150, λ_sym = 0.25,
                       tol_agg = 1e-5, maxiter_agg = 40, λ_agg = 0.4,
                       n_sims = 2_000, n_periods = 300, burnin = 100,
                       seed = 20260730)
    par = Params(n = n, β = β, σ = σ, μ = μ, γ = γ, θ = θ, ε = ε, η̄ = η̄)
    set = Settings(gmin = amin, gmax = amax, kx = k, ky = k,
                   spacing = spacing, spacing_param = spacing_param,
                   yspacing = spacing, yspacing_param = spacing_param,
                   tol_vfi = tol_vfi, maxiter_vfi = maxiter_vfi,
                   tol_sym_policy = tol_sym, maxiter_sym_policy = maxiter_sym,
                   λ_sym_policy = λ_sym,
                   tol_agg = tol_agg, maxiter_agg = maxiter_agg, λ_agg = λ_agg,
                   g_w0 = g_w, g_y0 = g_y, ŷ0 = ŷ,
                   n_sims = n_sims, n_periods = n_periods, burnin = burnin,
                   seed = seed, verbose = false)
    return DSIC(params = par, settings = set)
end

# =====================================================================
#  2. Solving
# =====================================================================

"""
    solve_model!(model, ws, draws; progress = true) -> (status, trace)

Run the aggregate loop, collecting a row per pass.

The progress bar tracks the *aggregate* loop only. The inner loops run
hundreds of iterations each and would fight it for the same terminal line;
their iteration counts go in the trace instead, which is the more useful
signal — if the value iteration is still taking hundreds of sweeps on the
tenth aggregate pass, the warm start has stopped working.
"""
function solve_model!(model::DSIC, ws::VFIWorkspace, draws::SimulationDraws;
                      progress::Bool = true)
    set = model.settings
    Λ = contraction_modulus(model.params, model.sol.aggs)

    if progress
        @printf("Λ = β(1+g_y)^-σ·γ = %.4f %s   (floor on g_y: %.5f)\n",
                Λ, Λ < 1 ? "✓" : "✗", feasible_growth_floor(model.params))
        @printf("n = %d   state space %d × %d = %d\n\n", model.params.n,
                size(model.sol.V)..., length(model.sol.V))
    end

    trace = NamedTuple[]
    bar = progress ? ProgressBar(set.maxiter_agg, set.tol_agg) : nothing

    st = solve_equilibrium!(model, ws, draws;
        on_iter = (iter, resid, res) -> begin
            push!(trace, (iter = iter, resid = resid, g_w = res.g_w,
                          ŷ = res.ŷ, L_r = res.L_r, ℒ = res.ℒ,
                          n_active = res.n_active, outside = res.outside_frac,
                          sym = model.sol.sym_policy.iters,
                          vfi = model.sol.vfi.iters))
            bar === nothing || update!(bar, resid, iter)
        end)

    if progress
        st.converged ? finish!(bar, st.iters, st.residual) :
            println("\n✗ stopped after $(st.iters) passes, residual $(st.residual)")
    end
    return (st, trace)
end

# =====================================================================
#  3. Panels
# =====================================================================

"""
    Panel

A pooled cross-section. Industry-level vectors have one entry per recorded
industry-period; firm-level vectors have `n` times as many.

Industry: `lerner` (`∑ᵢℓᵢ`), `n_active`, `a_tilde`, `price`, `gap`
(leader over laggard).

Firm: `rel_pos` (`aᵢ/ã`), `research` (`l`), `eta` (`η`), `share` (`sᵢ`).
"""
struct Panel
    lerner::Vector{Float64}
    n_active::Vector{Int}
    a_tilde::Vector{Float64}
    price::Vector{Float64}
    gap::Vector{Float64}
    rel_pos::Vector{Float64}
    research::Vector{Float64}
    eta::Vector{Float64}
    share::Vector{Float64}
end

Panel() = Panel(Float64[], Int[], Float64[], Float64[], Float64[],
                Float64[], Float64[], Float64[], Float64[])

Base.length(p::Panel) = length(p.lerner)

Base.show(io::IO, p::Panel) =
    print(io, "Panel(", length(p.lerner), " industry-periods, ",
              length(p.rel_pos), " firm-periods)")

"""
    simulate_panel(model, draws; a_init = nothing, thin = 5) -> Panel

Run the economy under the converged policy and keep the cross-section.

Records every `thin`-th post-burn-in period. Pass `a_init` to start from a
different cross-section; the level is renormalised first either way, so
only the shape of what you pass matters.
"""
function simulate_panel(model::DSIC, draws::SimulationDraws;
                        a_init = nothing, thin::Int = 5)
    N = state_length(model.params)
    return _panel(model, draws, a_init, thin, Val(N))
end

function _panel(model::DSIC, draws::SimulationDraws, a_init, thin::Int,
                ::Val{N}) where {N}
    par, set, sol = model.params, model.settings, model.sol
    μ, γ = par.μ, par.γ
    haz  = Hazard(par.η̄, par.θ, par.ε)
    π̃    = Interpolant(sol.policy, model.grid)
    S    = set.n_sims

    A = a_init === nothing ? copy(draws.a0) : copy(a_init)
    A ./= GE._level_factor(A, μ, S, Val(N))

    p = Panel()

    for t in 1:set.n_periods
        record = t > set.burnin && (t - set.burnin) % thin == 0

        for s in 1:S
            if record
                profile = GE.firm_view(A, s, 1, Val(N))
                st   = GE.industry_stats(profile, μ, Val(N))
                meta = market_share_with_meta(profile, μ)
                push!(p.lerner, st.lerner_sum)
                push!(p.n_active, st.n_active)
                push!(p.a_tilde, meta.a_tilde)
                push!(p.price, meta.p)

                lo = hi = A[s, 1]
                for i in 1:N
                    a_i = A[s, i]
                    lo = min(lo, a_i); hi = max(hi, a_i)
                    push!(p.rel_pos, a_i / meta.a_tilde)
                    push!(p.share, max(0.0, μ * (1 - meta.a_tilde / (meta.m * a_i))))
                end
                push!(p.gap, hi / lo)
            end

            for i in 1:N
                v = GE.firm_view(A, s, i, Val(N))
                l = max(π̃(v), 0.0)
                η = innovation_prob(l, v[1], haz)
                if record
                    push!(p.research, l)
                    push!(p.eta, η)
                end
                draws.u[s, i, t] < η && (A[s, i] *= γ)
            end
        end

        A ./= GE._level_factor(A, μ, S, Val(N))   # this is what defines w
    end

    return p
end

"""
    symmetric_start_panel(model, draws; thin = 5) -> Panel

The same economy, but every industry begins with all `n` firms at identical
productivity — the geometric middle of the grid. They come apart only
because some draw a successful innovation and others do not.
"""
function symmetric_start_panel(model::DSIC, draws::SimulationDraws;
                               thin::Int = 5)
    N  = state_length(model.params)
    S  = model.settings.n_sims
    lo, hi = extrema(xaxis(model.grid))
    a0 = sqrt(lo * hi)
    return simulate_panel(model, draws; a_init = fill(a0, S, N), thin = thin)
end

"""
    binned_mean(x, y; nbins = 25, lo, hi) -> (centres, means, counts)

Average `y` within equal-width bins of `x`. Empty bins come back as `NaN`
so a line plot breaks rather than interpolating across a gap.
"""
function binned_mean(x::AbstractVector, y::AbstractVector; nbins::Int = 25,
                     lo = quantile(x, 0.01), hi = quantile(x, 0.99))
    edges   = range(lo, hi; length = nbins + 1)
    centres = [(edges[i] + edges[i + 1]) / 2 for i in 1:nbins]
    sums    = zeros(nbins)
    counts  = zeros(Int, nbins)
    w = (hi - lo) / nbins
    for (xi, yi) in zip(x, y)
        (lo <= xi <= hi) || continue
        b = clamp(floor(Int, (xi - lo) / w) + 1, 1, nbins)
        sums[b] += yi
        counts[b] += 1
    end
    means = [counts[i] > 0 ? sums[i] / counts[i] : NaN for i in 1:nbins]
    return (centres, means, counts)
end

# =====================================================================
#  4. Result and report
# =====================================================================

"""
    ModelResult

Everything one run produced: the solved `model`, the `status` of the
aggregate loop, the per-pass `trace`, and the two panels. `ws` and `draws`
are kept so a follow-up run can reuse them.
"""
struct ModelResult{W,D}
    model::DSIC
    ws::W
    draws::D
    status::LoopStatus
    trace::Vector{NamedTuple}
    panel::Panel
    sym_panel::Panel
end

Base.show(io::IO, r::ModelResult) = @printf(io,
    "ModelResult(n=%d, %s, g_w=%.5f, ŷ=%.4f, ℒ=%.4f)",
    r.model.params.n, r.status.converged ? "converged" : "NOT converged",
    r.model.sol.aggs.g_w, r.model.sol.aggs.ŷ, r.model.sol.ℒ)

qtiles(x) = (quantile(x, 0.10), median(x), quantile(x, 0.90))

function describe(name, x; d = 4)
    q10, q50, q90 = qtiles(x)
    @printf("%-24s mean %8.*f  sd %8.*f  [p10 %7.*f  p50 %7.*f  p90 %7.*f]\n",
            name, d, mean(x), d, std(x), d, q10, d, q50, d, q90)
end

"""
    report(res::ModelResult)

Print the aggregates, the industry cross-section, and the within-industry
spread. The two `aᵢ/ã` lines — one from each panel — should agree; they
start from opposite ends of the state space and only the ergodic
distribution is common to both.
"""
function report(res::ModelResult)
    m, sol, par = res.model, res.model.sol, res.model.params
    st, p, sp = res.status, res.panel, res.sym_panel

    println("\n", "═"^90)
    println("AGGREGATES", st.converged ? "  ✓ converged" : "  ✗ NOT converged",
            "  after $(st.iters) passes, residual ",
            round(st.residual, sigdigits = 3))
    println("═"^90)
    @printf("%-24s %10.6f\n", "g_w = g_y", sol.aggs.g_w)
    @printf("%-24s %10.6f\n", "ŷ", sol.aggs.ŷ)
    @printf("%-24s %10.6f\n", "L^r   research labour", sol.L_r)
    @printf("%-24s %10.6f\n", "ℒ     Lerner index", sol.ℒ)
    @printf("%-24s %10.6f   (identity check)\n", "(1-L^r)/(1-ℒ)",
            (1 - sol.L_r) / (1 - sol.ℒ))
    if !isempty(res.trace)
        t = res.trace[end]
        @printf("%-24s %10.3f%%  ← policy extrapolated past the grid\n",
                "states outside grid", 100 * t.outside)
        @printf("%-24s %10d / %d\n", "sym-policy iters (last)", t.sym,
                m.settings.maxiter_sym_policy)
        @printf("%-24s %10d / %d\n", "vfi sweeps (last)", t.vfi,
                m.settings.maxiter_vfi)
    end

    println("\n", "─"^90)
    println("INDUSTRY CROSS-SECTION   (", length(p), " industry-periods)")
    println("─"^90)
    describe("Lerner  ∑ℓᵢ", p.lerner)
    describe("HHI  = μ·∑ℓᵢ", par.μ .* p.lerner)
    describe("ã  industry prod.", p.a_tilde)
    describe("p  relative price", p.price)
    for k in 1:par.n
        @printf("%-24s %10.4f\n", "share with ñ = $k",
                count(==(k), p.n_active) / length(p.n_active))
    end

    println("\n", "─"^90)
    println("FIRMS")
    println("─"^90)
    describe("market share  sᵢ", p.share)
    describe("research  l", p.research; d = 6)
    describe("innovation prob  η", p.eta)

    println("\n", "─"^90)
    println("WITHIN INDUSTRY   (all firms start level)")
    println("─"^90)
    describe("aᵢ/ã   symmetric start", sp.rel_pos)
    describe("aᵢ/ã   ergodic start", p.rel_pos)
    describe("leader / laggard", sp.gap)
    @printf("%-24s %10.4f\n", "P(gap > 1.5)",
            count(>(1.5), sp.gap) / length(sp.gap))
    @printf("%-24s %10.4f\n", "P(a firm shut out)",
            count(<(par.n), sp.n_active) / length(sp.n_active))
    println("═"^90)
    return nothing
end

# =====================================================================
#  5. One call
# =====================================================================

"""
    run_model(; thin = 5, progress = true, verbose = true, kwargs...) -> ModelResult

Build, solve, simulate, report. Keywords go to [`build_model`](@ref).

```julia
res = run_model(n = 2, η̄ = 2.0)
res = run_model(n = 3, k = 40, ε = 0.9)
plot_results(res)
```
"""
function run_model(; thin::Int = 5, progress::Bool = true,
                     verbose::Bool = true, kwargs...)
    model = build_model(; kwargs...)
    ws    = VFIWorkspace(model)
    draws = SimulationDraws(model)

    status, trace = solve_model!(model, ws, draws; progress = progress)

    panel = simulate_panel(model, draws; thin = thin)
    sym   = symmetric_start_panel(model, draws; thin = thin)

    res = ModelResult(model, ws, draws, status, trace, panel, sym)
    verbose && report(res)
    return res
end

end # module