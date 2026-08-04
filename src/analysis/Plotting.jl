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

    sim = run_simulation(res.model, SimSettings())
    plot_stationarity(sim; save = "output/baseline")
    plot_industry(sim;     save = "output/baseline")

Five pictures of four different things:

  * [`plot_results`](@ref) — the ergodic distribution, pooled over periods.
    What the economy looks like.
  * [`plot_paths`](@ref) — aggregates period by period after burn-in, under
    the converged policy. How the economy fluctuates along a sample path.
  * [`plot_convergence`](@ref) — one point per aggregate-loop iteration.
    How the *solver* got there; nothing to do with the economics.
  * [`plot_stationarity`](@ref) and [`plot_industry`](@ref) — the same two
    questions asked of a standalone [`run_simulation`](@ref) rather than of
    a `ModelResult`, so a solved model can be simulated at several sample
    sizes and each plotted on its own.

# Saving

The first three take `save` as a plain filename and hand it to `savefig`.
The two simulation plots take `save` as a *directory-plus-prefix* and write
a timestamped pair — the figure and a `.toml` of the whole run
configuration under the identical name — so a figure on disk can always be
traced back to the parameters that produced it.
"""
module Plotting

using Printf, Statistics, Plots

using ..StateGrids
using ..DSICModel
using ..StaticMarket
using ..ResearchPolicyFOC
using ..Runner
using ..Simulation

export plot_results, plot_paths, plot_convergence, policy_surface,
       plot_policy, plot_stationarity, plot_industry

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
    plot_paths(m::DSIC, p::Panel; save = nothing, extra_text = "") -> Plot
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

The two-argument form takes a bare `Panel`, so a panel simulated by hand can
be plotted without a `ModelResult` around it. Set `panel = :symmetric` on
the `ModelResult` form to trace the run that starts every industry level
instead. Requires `thin = 1` in `run_model` for a dense path.
"""
function plot_paths(m::DSIC, p::Panel; save = nothing, extra_text::String = "")
    par = m.params

    npanel_periods(p) > 1 || throw(ArgumentError(
        "the panel holds $(npanel_periods(p)) recorded period(s); re-run with " *
        "thin = 1 (and burnin < n_periods) to trace a path"))

    ln  = (; legend = false, grid = true, lw = 1.6)
    eq  = (; ls = :dash, c = :red, lw = 1.5)

    # --- aggregates the simulation computes directly -----------------
    p1 = plot(p.periods, p.g_w * 100; title = "Wage growth", ylabel = "\$g^w\$ (%)", ln...)
    hline!(p1, [m.sol.aggs.g_w * 100]; eq...)

    p2 = plot(p.periods, p.L_r * 100; title = "Research labour",
              ylabel = "\$L^r\$ (%)", ln...)
    hline!(p2, [m.sol.L_r * 100]; eq...)

    p3 = plot(p.periods, p.scriptL * 100; title = "Lerner Index  ℒ",
              ylabel = "ℒ (%)", ln...)
    hline!(p3, [m.sol.ℒ * 100]; eq...)

    p4 = plot(p.periods, p.yhat; title = "Demand shifter  \$\\hat{y}\$",
              ylabel = "\$\\hat{y}\$", ln...)
    hline!(p4, [m.sol.aggs.ŷ]; eq...)

    # --- market structure: every ñ in one panel ----------------------
    p5 = plot(; title = "Share of industries by \$\\tilde{n}\$", ylabel = "share",
                legend = :outerright, grid = true,
                ylims = (0, 100))
    for k in 1:par.n
        ts, sh = share_by_period(p.period, p.n_active, k)
        plot!(p5, ts, sh .* 100; lw = 1.6, label = "ñ = $k")
    end

    # --- cross-sectional moments: mean and median together -----------
    p6 = _moment_panel(p.fperiod, p.share, "Market share  \$s_i\$", "\$s_i\$"; xlabel = "")
    hline!(p6, [1 / par.n]; ls = :dot, c = :black, lw = 1, label = "")

    p7 = _moment_panel(p.period, p.lerner, "industry Lerner  \$\\sum \\ell_i\$", "\$\\ell\$")
    hline!(p7, [m.sol.ℒ], label = ""; eq...)

    p8 = _moment_panel(p.period, p.a_tilde, "Industry productivity  \$\\tilde{a}\$", "\$\\tilde{a}\$")

    p9 = _moment_panel(p.period, p.gap, "leader / laggard", "\$A_{\\max}/A_{\\min}\$"; yscale = :log10)
    hline!(p9, [1.0]; ls = :dot, c = :black, lw = 1, label = "Aₘₐₓ=Aₘᵢₙ")

    ttl = _title(m) * extra_text

    plt = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9; layout = (3, 3),
               size = (1600, 1050), plot_title = ttl, plot_titlefontsize = 9)

    save === nothing || savefig(plt, save)
    return plt
end

function plot_paths(res::ModelResult; save = nothing, panel::Symbol = :ergodic)
    panel in (:ergodic, :symmetric) || throw(ArgumentError(
        "panel must be :ergodic or :symmetric, got $(repr(panel))"))
    p = panel === :ergodic ? res.panel : res.sym_panel
    m = res.model
    return plot_paths(m, p; save = save,
                      extra_text = @sprintf("   |   %s start, %d periods",
                                            panel === :ergodic ? "ergodic" : "symmetric",
                                            npanel_periods(p)))
end

"""
    _moment_panel(labels, x, title, ylabel; kwargs...) -> Plot

Mean and median of `x` period by period, on one axis. The two together say
more than either alone: their gap is the skewness of the cross-section, and
a widening gap means the distribution is stretching even when its centre is
still.

`kwargs` go to the first `plot` call, where the axis is created, so an
`xlabel` or `yscale` passed in overrides the defaults set here rather than
being applied after the fact.
"""
function _moment_panel(labels, x, title, ylab; kwargs...)
    ts, avg = by_period(labels, x, mean)
    _,  med = by_period(labels, x, median)
    p = plot(ts, avg; label = "mean", lw = 1.6, c = 1,
             title = title, xlabel = "period", ylabel = ylab,
             legend = :best, grid = true, kwargs...)
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
#  Simulation output
#
#  These take a `SimulationOutput` rather than a `ModelResult`: the
#  simulation is run separately from the solve, so a solved model can be
#  simulated at several sample sizes and each plotted on its own.
#
#  `save = "output/baseline"` writes
#     output/baseline_<tag>_<timestamp>.png
#     output/baseline_<tag>_<timestamp>.toml
#  the second holding the full run configuration, so a figure on disk can
#  always be traced back to the parameters that produced it.
# =====================================================================

"""
    _save_figure(plt, save, tag, sim) -> String

Write `plt` and its configuration under one timestamped stem. The stem is
computed **once** so the two files pair exactly; calling `artifact_stem`
twice would stamp different seconds and silently separate them.
"""
function _save_figure(plt, save, tag::AbstractString, sim::SimulationOutput)
    stem = artifact_stem(save, tag)
    savefig(plt, stem * ".png")
    write_run_config(stem * ".toml", sim)
    @info "saved figure" figure = stem * ".png" config = stem * ".toml"
    return stem
end

"""
    plot_stationarity(sim::SimulationOutput; save = nothing) -> Plot

Seven stacked panels of the aggregates over time, each with its own mean
drawn in — a stationarity check.

What you are looking for is fluctuation *around* the dashed line with no
trend. A drift means burn-in ended too early and the panel is still
travelling toward its ergodic distribution, in which case every
cross-sectional moment reported elsewhere is contaminated.

Productivity indices are divided by `𝓜 = 1/(1-𝓛)` to convert from wage
units to baseline units — the bridge `a = aʷ/𝓜`.

Pass `save = "output/baseline"` to write the figure and a matching `.toml`
of the run configuration.
"""
function plot_stationarity(sim::SimulationOutput; save = nothing)
    a  = sim.agg
    n  = sim.params.n
    ℳ = mean(a.markup)

    ln = (; legend = false, lw = 1.6, grid = true, xlabel = "")
    avg(x) = hline!([mean(x)]; ls = :dash, c = :red, lw = 1.5)

    p1 = plot(a.period, a.g_w * 100;   title = "Growth rate \$g^w\$ (%)",       ln...); avg(a.g_w * 100)
    p2 = plot(a.period, a.L_r * 100;   title = "Research Labour \$L^r\$ (%)",   ln...); avg(a.L_r * 100)
    p3 = plot(a.period, a.scriptL;     title = "Lerner index \$\\mathcal{L}\$", ln...); avg(a.scriptL)
    p4 = plot(a.period, a.markup;      title = "Markup \$\\mathcal{M}\$",       ln...); avg(a.markup)
    p5 = plot(a.period, a.A_star ./ ℳ; title = "Frontier index \$a^*\$",       ln...); avg(a.A_star ./ ℳ)

    # the two wedges and their product
    p6 = plot(a.period, a.Lambda_w * 100; label = "\$\\Lambda^w\$", lw = 1.6,
              grid = true, title = "Efficiency Wedges (%)", xlabel = "",
              legend = :best)
    plot!(p6, a.period, a.Lambda_x * 100; label = "\$\\Lambda^x\$", lw = 1.6)
    plot!(p6, a.period, a.Lambda_w .* a.Lambda_x * 100;
          label = "\$\\Lambda\$", lw = 2, c = :black)

    # active-firm shares, all ñ in one panel
    p7 = plot(; title = "Share of industries by \$\\tilde{n}\$",
                xlabel = "period", ylabel = "share", ylims = (0, 100),
                legend = :best, grid = true)
    for k in 1:n
        plot!(p7, a.period, a.n_share[:, k] * 100; lw = 1.6, label = "ñ = $k")
    end

    plt = plot(p1, p2, p3, p4, p5, p6, p7; layout = (7, 1), size = (1000, 1500))
    save === nothing || _save_figure(plt, save, "stationarity", sim)
    return plt
end

"""
    plot_industry(sim::SimulationOutput; save = nothing) -> Plot

Six panels of the industry cross-section: the distributions of the Lerner
index, the markup, the productive wedge and the frontier, then the mean and
median of `ℓ(j)` and `Δ(j)` over time.

The mean and median together are worth more than either alone — their gap
is the skewness of the cross-section, so a widening gap means the
distribution is stretching even when its centre is still.

`Δ(j) ≥ 1` always, with equality when only frontier firms produce, so the
mass to the right of the black line is the within-industry efficiency loss.

Requires `store_industry = true` in the `SimSettings`.
"""
function plot_industry(sim::SimulationOutput; save = nothing)
    a, ind = sim.agg, sim.industry
    size(ind, 1) > 0 || throw(ArgumentError(
        "this run stored no industry paths; re-run with " *
        "SimSettings(store_industry = true)"))
    ℳ = mean(a.markup)

    h = (; normalize = :probability, bins = 40, grid = true,
           ylabel = "probability")

    q1 = histogram(vec(ind.lerner); title = "Industry Lerner  \$\\ell(j)\$",
                   xlabel = "\$\\ell\$", label = "", h...)
    vline!(q1, [mean(a.scriptL)]; ls = :dash, c = :red, lw = 2,
           label = "\$\\mathcal{L}\$=$(round(mean(a.scriptL), digits = 3))")

    q2 = histogram(vec(ind.markup); title = "Industry Markup  \$\\mathcal{M}(j)\$",
                   xlabel = "\$\\mathcal{M}\$", label = "", h...)
    vline!(q2, [mean(a.markup)]; ls = :dash, c = :red, lw = 2,
           label = "\$\\mathcal{M}\$=$(round(mean(a.markup), digits = 3))")

    q3 = histogram(vec(ind.delta);
                   title = "Productive Wedge  \$\\Delta(j) = a_{\\max}/\\bar{a}\$",
                   xlabel = "\$\\Delta\$", label = "", h...)
    vline!(q3, [1.0]; ls = :dash, c = :black, lw = 1.5,
           label = "\$\\Delta=1\$ (no wedge)")
    vline!(q3, [mean(vec(ind.delta))]; ls = :dash, c = :red, lw = 2,
           label = "\$\\mathrm{avg}(\\Delta)\$=$(round(mean(vec(ind.delta)), digits = 3))")

    q4 = histogram(vec(ind.a_max ./ ℳ); title = "Frontier  \$a_{\\max}(j)\$",
                   xlabel = "\$A_{\\max}\$", label = "", h...)

    # time paths of the cross-sectional mean and median
    med(x) = [median(view(x, t, :)) for t in 1:size(x, 1)]
    mn(x)  = [mean(view(x, t, :))   for t in 1:size(x, 1)]

    q5 = plot(a.period, mn(ind.lerner); label = "mean", lw = 1.6,
              title = "\$\\ell(j)\$ over time", xlabel = "period",
              ylabel = "\$\\ell\$", grid = true)
    plot!(q5, a.period, med(ind.lerner); label = "median", lw = 1.6,
          ls = :dashdot)

    q6 = plot(a.period, mn(ind.delta); label = "mean", lw = 1.6,
              title = "\$\\Delta(j)\$ over time", xlabel = "period",
              ylabel = "\$\\Delta\$", grid = true)
    plot!(q6, a.period, med(ind.delta); label = "median", lw = 1.6,
          ls = :dashdot)

    plt = plot(q1, q2, q3, q4, q5, q6; layout = (3, 2), size = (1000, 1000))
    save === nothing || _save_figure(plt, save, "industry", sim)
    return plt
end

# =====================================================================
function _title(m::DSIC)
    par = m.params
    txt =  @sprintf("n=%d  γ=%.3f  θ=%.2f  ε=%.2f  η̄=%.2f  μ=%.2f  β=%.2f   |   \$\\hat{y}\$=%.3f",
                    par.n, par.γ, par.θ, par.ε, par.η̄, par.μ, par.β,
                    m.sol.aggs.ŷ)
    for (t, v) in [
        ("\$g^w\$", m.sol.aggs.g_w * 100),
        ("\$L^r\$", m.sol.L_r * 100),
        ("ℒ", m.sol.ℒ * 100)
    ]
        txt *= (@sprintf("   %s=%.2f", t, v) * "%")
    end
    return txt
end

_title(res::ModelResult) = _title(res.model)

end # module