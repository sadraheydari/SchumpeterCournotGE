"""
    Plotting

Pictures of a solved model. Kept in its own submodule because it is the
only thing in the package that needs `Plots`, which dominates the load
time — comment out its `include` in `SchumpeterCournotGE` if you are
running headless.

    res = run_model(n = 2)
    plot_results(res)      # cross-sections and firm behaviour
    plot_paths(res)        # the sample path, period by period
    plot_convergence(res)  # the solver's own convergence

Three different pictures of three different things:

  * [`plot_results`](@ref) — the ergodic distribution, pooled over periods.
    What the economy looks like.
  * [`plot_paths`](@ref) — aggregates period by period after burn-in, under
    the converged policy. How the economy fluctuates along a sample path.
  * [`plot_convergence`](@ref) — one point per aggregate-loop iteration.
    How the *solver* got there; nothing to do with the economics.
"""
module Plotting

using Printf, Statistics, Plots

using ..StateGrids
using ..DSICModel
using ..StaticMarket
using ..ResearchPolicyFOC
using ..Runner

export plot_results, plot_paths, plot_convergence, policy_surface, plot_policy

# =====================================================================
#  The policy surface
# =====================================================================

"""
    policy_surface(model; others = nothing, kind = :policy) -> (xs, ys, Z)

The research policy over two productivity dimensions: the firm's own on the
`x` axis, one rival's on the `y` axis.

With more than two firms the remaining rivals have to be held somewhere.
`others` sets that level, defaulting to the geometric middle of the grid.
It is a *slice*, not a marginal — the surface moves as `others` moves, and
with strong catch-up (`ε` large) it moves a lot.

`kind` selects `:policy` (research `l`) or `:value` (the value function).
"""
function policy_surface(model::DSIC; others = nothing, kind::Symbol = :policy)
    N = state_length(model.params)
    N >= 2 || throw(ArgumentError(
        "a surface needs at least two productivity dimensions; n = $N has no rival"))
    kind in (:policy, :value) ||
        throw(ArgumentError("kind must be :policy or :value, got $(repr(kind))"))

    arr = kind === :policy ? model.sol.policy : model.sol.V
    f   = Interpolant(arr, model.grid)
    xs  = nodes(xaxis(model.grid))
    ys  = nodes(yaxis(model.grid))

    lo, hi = extrema(xaxis(model.grid))
    rest   = others === nothing ? sqrt(lo * hi) : Float64(others)

    Z = [f((x, y, ntuple(_ -> rest, Val(N - 2))...)) for y in ys, x in xs]
    return (xs, ys, Z)
end

"""
    plot_policy(model; others = nothing, kind = :policy) -> Plot

Heatmap of [`policy_surface`](@ref), with the 45° line marking where the
firm and the plotted rival are level. Above it the firm leads; below, it
trails.
"""
function plot_policy(model::DSIC; others = nothing, kind::Symbol = :policy,
                     title = nothing)
    xs, ys, Z = policy_surface(model; others = others, kind = kind)
    n = model.params.n
    ttl = title !== nothing ? title :
          (kind === :policy ? "research  l(a_own, a_rival)" :
                              "value  V(a_own, a_rival)")
    n > 2 && (ttl *= @sprintf("   [%d others at %.2f]", n - 2,
                              others === nothing ?
                              sqrt(prod(extrema(xaxis(model.grid)))) : others))

    p = heatmap(xs, ys, Z; title = ttl, xlabel = "a_own", ylabel = "a_rival",
                c = :viridis, colorbar = true)
    lo, hi = extrema(xs)
    plot!(p, [lo, hi], [lo, hi]; c = :white, ls = :dash, lw = 1.2, label = "")
    return p
end

# =====================================================================
#  Cross-sections
# =====================================================================

"""
    plot_results(res::ModelResult; save = nothing, others = nothing) -> Plot

Eight panels of the ergodic distribution, pooled over all recorded periods.

**Firm behaviour** — the policy surface; research by within-industry
position; innovation probability by position; the distribution of market
shares.

Research and innovation *by position* are the escape-competition picture:
whether a firm that has fallen behind tries harder to catch up or gives up,
read off the slope, and whether a leader coasts.

**Industry outcomes** — the Lerner coefficient `∑ℓᵢ` with the aggregate
`ℒ` marked; how many firms survive; industry productivity `ã`; and the
leader-to-laggard gap from industries that began level.
"""
function plot_results(res::ModelResult; save = nothing, others = nothing,
                      nbins::Int = 30)
    m, par = res.model, res.model.params
    p, sp  = res.panel, res.sym_panel

    h = (; legend = false, grid = true, normalize = :pdf, lw = 0.4,
           bins = nbins * 2)
    ln = (; legend = false, grid = true, lw = 2, marker = :circle, ms = 2.5)

    # --- firm behaviour ---------------------------------------------
    p1 = par.n >= 2 ? plot_policy(m; others = others) :
         plot(nodes(xaxis(m.grid)), vec(m.sol.policy.data);
              title = "research  l(a)", xlabel = "a", ylabel = "l",
              legend = false, lw = 2, grid = true)

    xc, yl, _ = binned_mean(p.rel_pos, p.research; nbins = nbins)
    p2 = plot(xc, yl; title = "research by position", xlabel = "aᵢ / ã",
              ylabel = "mean l", ln...)
    vline!(p2, [1.0]; ls = :dash, c = :black, lw = 1)

    xc2, ye, _ = binned_mean(p.rel_pos, p.eta; nbins = nbins)
    p3 = plot(xc2, ye; title = "innovation probability by position",
              xlabel = "aᵢ / ã", ylabel = "mean η", ln...)
    vline!(p3, [1.0]; ls = :dash, c = :black, lw = 1)

    p4 = histogram(p.share; title = "market share  sᵢ", xlabel = "sᵢ",
                   ylabel = "density", h...)
    vline!(p4, [1 / par.n]; ls = :dash, c = :red, lw = 2)

    # --- industry outcomes ------------------------------------------
    p5 = histogram(p.lerner; title = "industry Lerner  ∑ℓᵢ", xlabel = "ℓ",
                   ylabel = "density", h...)
    vline!(p5, [m.sol.ℒ]; ls = :dash, c = :red, lw = 2)

    shares = [count(==(k), p.n_active) / length(p.n_active) for k in 1:par.n]
    p6 = bar(1:par.n, shares; title = "active firms  ñ", xlabel = "ñ",
             ylabel = "share", xticks = 1:par.n, legend = false, grid = true)

    p7 = histogram(p.a_tilde; title = "industry productivity  ã",
                   xlabel = "ã", ylabel = "density", h...)

    p8 = histogram(sp.gap; title = "leader / laggard  (level start)",
                   xlabel = "a_max / a_min", ylabel = "density", h...)
    vline!(p8, [1.0]; ls = :dash, c = :black, lw = 1)

    plt = plot(p1, p2, p3, p4, p5, p6, p7, p8; layout = (2, 4),
               size = (1800, 880), plot_title = _title(res),
               plot_titlefontsize = 9)

    save === nothing || savefig(plt, save)
    return plt
end

# =====================================================================
#  Sample paths
# =====================================================================

"""
    plot_paths(res::ModelResult; save = nothing, panel = :ergodic) -> Plot

Nine panels tracing the economy period by period after burn-in, under the
converged policy.

**Aggregates** — `g_w`, `L^r`, `𝓛`, `ŷ`. Each carries a dashed red line at
the equilibrium value the outer loop settled on. The series should
fluctuate around it without drifting: a trend means burn-in was too short,
and a level offset means the simulation the paths come from is not the one
that closed the loop.

**Market structure** — the share of industries with `ñ = 1, …, n`, all in
one panel, so entry and exit can be read against each other.

**Cross-sectional moments** — mean and median together, for market share,
the industry Lerner coefficient, `ã`, and the leader-to-laggard gap. The
gap between mean and median is the skewness of the cross-section, and it
moves: a widening gap in the Lerner coefficient means market power is
concentrating in fewer industries even when the average is flat.

Set `panel = :symmetric` to trace the run that starts every industry level
instead. Requires `thin = 1` in `run_model` for a dense path.
"""
function plot_paths(res::ModelResult; save = nothing, panel::Symbol = :ergodic)
    panel in (:ergodic, :symmetric) || throw(ArgumentError(
        "panel must be :ergodic or :symmetric, got $(repr(panel))"))
    p   = panel === :ergodic ? res.panel : res.sym_panel
    m   = res.model
    par = m.params

    npanel_periods(p) > 1 || throw(ArgumentError(
        "the panel holds $(npanel_periods(p)) recorded period(s); re-run with " *
        "thin = 1 (and burnin < n_periods) to trace a path"))

    ln  = (; legend = false, grid = true, lw = 1.6, xlabel = "period")
    eq  = (; ls = :dash, c = :red, lw = 1.5)

    # --- aggregates the simulation computes directly -----------------
    p1 = plot(p.periods, p.g_w; title = "wage growth  g_w", ylabel = "g_w", ln...)
    hline!(p1, [m.sol.aggs.g_w]; eq...)

    p2 = plot(p.periods, p.L_r; title = "research labour  L^r",
              ylabel = "L^r", ln...)
    hline!(p2, [m.sol.L_r]; eq...)

    p3 = plot(p.periods, p.scriptL; title = "aggregate Lerner  ℒ",
              ylabel = "ℒ", ln...)
    hline!(p3, [m.sol.ℒ]; eq...)

    p4 = plot(p.periods, p.yhat; title = "demand shifter  ŷ",
              ylabel = "ŷ", ln...)
    hline!(p4, [m.sol.aggs.ŷ]; eq...)

    # --- market structure: every ñ in one panel ----------------------
    p5 = plot(; title = "share of industries by ñ", ylabel = "share",
                xlabel = "period", legend = :outerright, grid = true,
                ylims = (0, 1))
    for k in 1:par.n
        ts, sh = share_by_period(p.period, p.n_active, k)
        plot!(p5, ts, sh; lw = 1.6, label = "ñ = $k")
    end

    # --- cross-sectional moments: mean and median together -----------
    p6 = _moment_panel(p.fperiod, p.share, "market share  sᵢ", "sᵢ")
    hline!(p6, [1 / par.n]; ls = :dot, c = :black, lw = 1)

    p7 = _moment_panel(p.period, p.lerner, "industry Lerner  ∑ℓᵢ", "ℓ")
    hline!(p7, [m.sol.ℒ]; eq...)

    p8 = _moment_panel(p.period, p.a_tilde, "industry productivity  ã", "ã")

    p9 = _moment_panel(p.period, p.gap, "leader / laggard", "a_max/a_min")
    hline!(p9, [1.0]; ls = :dot, c = :black, lw = 1)

    ttl = _title(res) * @sprintf("   |   %s start, %d periods",
                                 panel === :ergodic ? "ergodic" : "symmetric",
                                 npanel_periods(p))
    plt = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9; layout = (3, 3),
               size = (1600, 1050), plot_title = ttl, plot_titlefontsize = 9)

    save === nothing || savefig(plt, save)
    return plt
end

"""
    _moment_panel(labels, x, title, ylabel) -> Plot

Mean and median of `x` period by period, on one axis. The two together say
more than either alone: their gap is the skewness of the cross-section, and
a widening gap means the distribution is stretching even when its centre is
still.
"""
function _moment_panel(labels, x, title, ylab)
    ts, avg = by_period(labels, x, mean)
    _,  med = by_period(labels, x, median)
    p = plot(ts, avg; label = "mean", lw = 1.6, c = 1,
             title = title, xlabel = "period", ylabel = ylab,
             legend = :best, grid = true)
    plot!(p, ts, med; label = "median", lw = 1.6, c = 2, ls = :dashdot)
    return p
end

# =====================================================================
#  Solver convergence
# =====================================================================

"""
    plot_convergence(res::ModelResult; save = nothing) -> Plot

The *solver's* path: one point per aggregate-loop iteration. This is
diagnostic, not economics — it says how the guessed aggregates travelled to
their fixed point, and how much work the inner loops needed on the way.

A residual that falls geometrically is healthy. One that plateaus means the
inner tolerances are too loose for the outer one; one that rises means
`λ_agg` is too large. The inner iteration counts should fall sharply after
the first pass — if they do not, warm starting has stopped working.
"""
function plot_convergence(res::ModelResult; save = nothing)
    isempty(res.trace) && throw(ArgumentError("the trace is empty"))
    t   = res.trace
    it  = [r.iter for r in t]
    set = res.model.settings

    ln = (; legend = false, grid = true, lw = 2, marker = :circle, ms = 3,
            xlabel = "aggregate pass")

    p1 = plot(it, max.([r.resid for r in t], 1e-16); yscale = :log10,
              title = "aggregate residual", ylabel = "‖Δ‖∞", ln...)
    hline!(p1, [set.tol_agg]; ls = :dash, c = :red)

    p2 = plot(it, [r.g_w for r in t]; title = "implied g_w", ylabel = "g_w", ln...)
    p3 = plot(it, [r.ℒ for r in t]; title = "implied ℒ", ylabel = "ℒ", ln...)
    p4 = plot(it, [r.L_r for r in t]; title = "implied L^r", ylabel = "L^r", ln...)

    p5 = plot(it, [r.sym for r in t]; title = "sym-policy iterations",
              ylabel = "iters", ln...)
    p6 = plot(it, [r.vfi for r in t]; title = "vfi sweeps",
              ylabel = "sweeps", ln...)

    p7 = plot(it, 100 .* [r.outside for r in t];
              title = "states outside grid", ylabel = "%", ln...)
    p8 = plot(it, [r.n_active for r in t]; title = "mean ñ", ylabel = "ñ", ln...)

    plt = plot(p1, p2, p3, p4, p5, p6, p7, p8; layout = (2, 4),
               size = (1700, 780), plot_title = _title(res),
               plot_titlefontsize = 9)

    save === nothing || savefig(plt, save)
    return plt
end

# =====================================================================

function _title(res::ModelResult)
    par, m = res.model.params, res.model
    return @sprintf("n=%d  γ=%.3f  θ=%.2f  ε=%.2f  η̄=%.2f  μ=%.2f  β=%.2f   |   g_w=%.5f  ŷ=%.4f  L^r=%.4f  ℒ=%.4f",
                    par.n, par.γ, par.θ, par.ε, par.η̄, par.μ, par.β,
                    m.sol.aggs.g_w, m.sol.aggs.ŷ, m.sol.L_r, m.sol.ℒ)
end

end # module