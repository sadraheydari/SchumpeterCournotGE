"""
    Plotting

Pictures of a solved model. Kept in its own submodule because it is the
only thing in the package that needs `Plots`, which dominates the load
time — comment out its `include` in `SchumpeterCournotGE` if you are
running headless.

    res = run_model(n = 2)
    plot_results(res)
    plot_results(res; save = "run.png")
"""
module Plotting

using Printf, Statistics, Plots

using ..StateGrids
using ..DSICModel
using ..StaticMarket
using ..ResearchPolicyFOC
using ..Runner

export plot_results, policy_surface, plot_policy

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
    N   = state_length(model.params)
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
#  The dashboard
# =====================================================================

"""
    plot_results(res::ModelResult; save = nothing, others = nothing) -> Plot

Eight panels.

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

    ttl = @sprintf("n=%d  γ=%.3f  θ=%.2f  ε=%.2f  η̄=%.2f  μ=%.2f  β=%.2f   |   g_w=%.3f  ŷ=%.4f  L^r=%.2f  ℒ=%.2f",
                   par.n, par.γ, par.θ, par.ε, par.η̄, par.μ, par.β,
                   m.sol.aggs.g_w * 100, m.sol.aggs.ŷ, m.sol.L_r * 100, m.sol.ℒ * 100)

    plt = plot(p1, p2, p3, p4, p5, p6, p7, p8; layout = (2, 4),
               size = (1800, 880), plot_title = ttl, plot_titlefontsize = 9)

    save === nothing || savefig(plt, save)
    return plt
end

end # module