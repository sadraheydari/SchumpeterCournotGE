"""
    Monopoly

The `n = 1` special case: one firm per industry, no rivals. Builds the model,
solves for the growth rate that clears the price index, and reads off the
stationary cross-section of industries.

Promoted out of `dev/monopoly_playgrowund.jl` (still there, and still the
place to explore a single calibration by hand) so that a batch sweep over
many calibrations — [`script/monopoly_sweep.jl`](@ref) — can call the same,
already-checked solving code rather than a copy of it.

# Reading order

 1. `build_monopoly` / `feasible_growth_floor` / `set_growth!` — build a model
    and move its growth rate around
 2. `stationary_distribution` / `boundary_mass` — the cross-section of
    industries implied by a solved model
 3. `implied_aggregates` — what that cross-section says about `g_w`, `ŷ`, etc.
 4. `level_residual!` / `find_g_w!` — bisect on the price-index identity that
    pins `g_w`; see the docstring on `find_g_w!` for why that equation and
    not the growth-consistency check
 5. `run_monopoly` — build, solve, find `g_w`, report
 6. `plot_monopoly` — the four-panel picture
"""
module Monopoly

using Printf, Plots

using ..SymStateArrays
using ..StateGrids
using ..DSICModel
using ..StaticMarket
using ..ResearchPolicyFOC
using ..ValueIteration

export build_monopoly, feasible_growth_floor, set_growth!,
       stationary_distribution, boundary_mass, implied_aggregates,
       level_residual!, find_g_w!, run_monopoly, plot_monopoly

# =====================================================================
#  1. Build and solve
# =====================================================================

"""
    build_monopoly(; kwargs...) -> DSIC

One firm per industry, on a grid over the de-trended productivity
`a = A/w`. `ky` is irrelevant when `n = 1` (there are no rivals) but the
grid still needs a `y` axis, so it is set equal to `kx`.
"""
function build_monopoly(; β = 0.96, σ = 2.0, μ = 2.0, γ = 1.05,
                          θ = 0.30, ε = 0.10, η̄ = 1.0,
                          amin = 0.20, amax = 6.0, kx = 200,
                          spacing = :power, spacing_param = 1.5,
                          g_w = 0.01, g_y = 0.01, ŷ = 2.0,
                          tol_vfi = 1e-11, maxiter_vfi = 5_000)
    par = Params(n = 1, β = β, σ = σ, μ = μ, γ = γ, θ = θ, ε = ε, η̄ = η̄)
    set = Settings(gmin = amin, gmax = amax, kx = kx, ky = kx,
                   spacing = spacing, spacing_param = spacing_param,
                   yspacing = spacing, yspacing_param = spacing_param,
                   tol_vfi = tol_vfi, maxiter_vfi = maxiter_vfi,
                   g_w0 = g_w, g_y0 = g_y, ŷ0 = ŷ,
                   n_periods = 650, burnin = 150, verbose = false)
    return DSIC(params = par, settings = set)
end

"""
    feasible_growth_floor(par) -> Float64

The smallest `g_y` for which the Bellman operator is a contraction.
`Λ = β(1+g_y)^{-σ}γ < 1` rearranges to `g_y > (βγ)^{1/σ} - 1`, so a model
with `γ > 1` has **no bounded solution at zero growth**. That is not a
numerical nuisance: if firms innovate forever and nothing grows to discount
it, the firm is worth infinity.
"""
feasible_growth_floor(par::Params) = (par.β * par.γ)^(1 / par.σ) - 1

"Replace the growth pair, keeping ŷ. On a balanced path `g_y = g_w`."
function set_growth!(model::DSIC, g::Real)
    model.sol.aggs = Aggregates(g, g, model.sol.aggs.ŷ)
    return model
end

# =====================================================================
#  2. The cross-section of industries
# =====================================================================

"""
    stationary_distribution(model; tol, maxit) -> (φ, iters, gap)

Push the cross-section forward until it stops moving.

Each industry sits at some `a`, innovates with probability `η(l(a), a)`
which multiplies productivity by `γ`, and either way divides by `(1+g_w)`.
The landing point is off-grid, so its mass is split between the two
neighbouring nodes by the same linear weights the interpolation uses.

**The mass is clamped into the grid, not extrapolated.** Extrapolation uses
negative weights — right for a value function, meaningless for a
distribution. The price is that mass piling on an edge signals too small a
grid, which `boundary_mass` reports.
"""
function stationary_distribution(model::DSIC; tol = 1e-14, maxit = 200_000)
    par, grid = model.params, model.grid
    ax   = xaxis(grid)
    a    = nodes(ax)
    k    = length(a)
    haz  = Hazard(par.η̄, par.θ, par.ε)
    g_w  = model.sol.aggs.g_w
    lo, hi = first(a), last(a)

    T = zeros(k, k)                       # row i: where a[i] goes next period
    for i in 1:k
        l = model.sol.policy.data[i, 1]
        η = innovation_prob(l, a[i], haz)
        for (prob, step) in ((η, par.γ), (1 - η, 1.0))
            prob == 0.0 && continue
            x    = clamp(step * a[i] / (1 + g_w), lo, hi)
            j, t = locate(ax, x)
            T[i, j]     += prob * (1 - t)
            T[i, j + 1] += prob * t
        end
    end

    φ = fill(1 / k, k)
    gap, iters = Inf, 0
    for it in 1:maxit
        φ_new = T' * φ
        gap   = maximum(abs, φ_new .- φ)
        φ, iters = φ_new, it
        gap < tol && break
    end
    return (φ, iters, gap)
end

"Share of the cross-section sitting on the two end nodes."
boundary_mass(φ) = (first(φ), last(φ))

# =====================================================================
#  3. What the cross-section implies
# =====================================================================

"""
    implied_aggregates(model, φ) -> NamedTuple

Read the aggregates back out of the stationary cross-section.

  * `level` — `∫p^{1-μ}dΦ`, which equilibrium requires to be exactly `1`.
    **This is the equation that determines `g_w`**; see the header.
  * `g_w` — the growth the productivity process implies. At stationarity
    this equals whatever `g_w` was assumed, identically, so it is a
    consistency check on the arithmetic and *not* an equilibrium residual.
  * `L_r` — `∫l dΦ`, since labour is one and there is a firm per industry.
  * `ℒ` — the output-share-weighted Lerner index. For `n = 1` it must come
    out as exactly `1/μ`, which checks `StaticMarket` rather than measuring
    anything.
"""
function implied_aggregates(model::DSIC, φ::Vector{Float64})
    par  = model.params
    a    = nodes(xaxis(model.grid))
    haz  = Hazard(par.η̄, par.θ, par.ε)
    μ, γ = par.μ, par.γ

    num_g = den_g = 0.0
    L_r   = η_avg = 0.0
    num_L = den_L = 0.0

    for i in eachindex(a)
        l  = model.sol.policy.data[i, 1]
        η  = innovation_prob(l, a[i], haz)
        wa = φ[i] * a[i]^(μ - 1)

        num_g += wa * (η * γ^(μ - 1) + (1 - η))
        den_g += wa
        L_r   += φ[i] * l
        η_avg += φ[i] * η

        num_L += φ[i] * operational_profit_weight((a[i],), μ)   # p^{1-μ}·ℓ
        den_L += φ[i] * industry_output_share((a[i],), μ)       # p^{1-μ}
    end

    g_w = (num_g / den_g)^(1 / (μ - 1)) - 1
    ℒ   = num_L / den_L

    return (g_w = g_w, g_y = g_w, L_r = L_r, ℒ = ℒ,
            ŷ = (1 - L_r) / (1 - ℒ),
            η_mean = η_avg, level = den_L, a_mean = sum(φ .* a))
end

# =====================================================================
#  4. Solving for g_w
# =====================================================================

"""
    level_residual!(model, ws, g; dist_tol) -> (residual, φ, implied)

Set the growth rate to `g`, re-solve the firm's problem, find the
stationary cross-section, and return `∫p^{1-μ}dΦ - 1`.

Positive means the distribution sits too high and `g` is too small.
The value function is **not** reset between calls, so each candidate warm
starts from the last — which is most of why the bisection is affordable.
"""
function level_residual!(model::DSIC, ws::VFIWorkspace, g::Real;
                         dist_tol = 1e-12)
    set_growth!(model, g)
    solve_vfi!(model, ws)
    φ, _, _ = stationary_distribution(model; tol = dist_tol)
    imp = implied_aggregates(model, φ)
    return (imp.level - 1.0, φ, imp)
end

"""
    find_g_w!(model, ws; kwargs...) -> NamedTuple

Bisect on the growth rate until `∫p^{1-μ}dΦ = 1`.

The residual is decreasing in `g` — a larger `g` divides harder, so mass
drifts down and the level falls — which makes bisection safe and means one
crossing. The bracket starts just above `feasible_growth_floor`, below
which the Bellman operator is not a contraction at all, and expands upward
until the residual turns negative.

Bisection rather than a secant method: each evaluation costs a full value
iteration plus a distribution solve, so robustness beats a few saved steps,
and the residual is only piecewise smooth once mass touches a boundary.
"""
function find_g_w!(model::DSIC, ws::VFIWorkspace;
                   tol = 1e-6, maxit = 40, verbose = true, dist_tol = 1e-12)
    par   = model.params
    floor_g = feasible_growth_floor(par)
    lo    = floor_g + 1e-3
    verbose && @printf("\nsolving for g_w   (contraction floor = %.6f)\n", floor_g)

    r_lo, _, _ = level_residual!(model, ws, lo; dist_tol = dist_tol)
    verbose && @printf("  g = %.6f   ∫p^{1-μ}-1 = %+.4e\n", lo, r_lo)
    if r_lo < 0
        @warn("even at the contraction floor the distribution sits too low; " *
              "no equilibrium growth rate exists for these parameters",
              floor = floor_g, residual = r_lo)
        return (g = lo, residual = r_lo, converged = false)
    end

    hi, r_hi = lo, r_lo
    for _ in 1:20                              # expand until it flips sign
        hi = hi + max(0.01, hi)
        r_hi, _, _ = level_residual!(model, ws, hi; dist_tol = dist_tol)
        verbose && @printf("  g = %.6f   ∫p^{1-μ}-1 = %+.4e\n", hi, r_hi)
        r_hi < 0 && break
    end
    if r_hi > 0
        @warn "could not bracket the root; the level stays above 1 for every g tried"
        return (g = hi, residual = r_hi, converged = false)
    end

    g, r = hi, r_hi
    for it in 1:maxit
        g = 0.5 * (lo + hi)
        r, _, _ = level_residual!(model, ws, g; dist_tol = dist_tol)
        r > 0 ? (lo = g) : (hi = g)
        verbose && @printf("  [%2d] g = %.8f   ∫p^{1-μ}-1 = %+.4e\n", it, g, r)
        abs(r) < tol && break
        (hi - lo) < 1e-12 && break
    end

    set_growth!(model, g)                      # leave the model at the answer
    solve_vfi!(model, ws)
    return (g = g, residual = r, converged = abs(r) < tol)
end

# =====================================================================
#  5. Everything at once
# =====================================================================

"""
    run_monopoly(; solve_growth = true, kwargs...) -> NamedTuple

Build, solve, find `g_w`, simulate, and report. Keywords go to
[`build_monopoly`](@ref). With `solve_growth = false` the guessed `g_w` is
used as-is, which is what you want when exploring by hand.
"""
function run_monopoly(; solve_growth = true, verbose = true, kwargs...)
    model = build_monopoly(; kwargs...)
    ws    = VFIWorkspace(model)

    st = solve_vfi!(model, ws)
    verbose && !st.converged &&
        println("✗ VFI stopped at $(st.iters) sweeps, residual $(st.residual)")
    gw = solve_growth ? find_g_w!(model, ws; verbose = verbose) : nothing

    φ, dit, dgap = stationary_distribution(model)
    imp   = implied_aggregates(model, φ)
    given = model.sol.aggs

    if verbose
        lo_m, hi_m = boundary_mass(φ)
        println()
        println("─"^70)
        @printf("distribution converged in %d steps (gap %.2e)\n", dit, dgap)
        @printf("mass on the boundaries:  low %.3e   high %.3e%s\n",
                lo_m, hi_m, max(lo_m, hi_m) > 1e-3 ? "   ← widen the grid" : "")
        println("─"^70)
        @printf("%-22s %12s %12s %12s\n", "", "given", "implied", "gap")
        @printf("%-22s %12.6f %12.6f %12.2e   ← THE equation for g_w\n",
                "∫p^{1-μ}", 1.0, imp.level, imp.level - 1)
        @printf("%-22s %12.6f %12.6f %12.2e\n", "ŷ", given.ŷ, imp.ŷ, imp.ŷ - given.ŷ)
        @printf("%-22s %12.6f %12.6f %12.2e   (identity — uninformative)\n",
                "g_w", given.g_w, imp.g_w, imp.g_w - given.g_w)
        println("─"^70)
        @printf("%-22s %12.6f   (floor %.6f)\n", "g_w = g_y", given.g_w,
                feasible_growth_floor(model.params))
        @printf("%-22s %12.6e\n", "L^r", imp.L_r)
        @printf("%-22s %12.6f   (1/μ = %.6f)\n", "ℒ", imp.ℒ, 1 / model.params.μ)
        @printf("%-22s %12.6f\n", "mean η", imp.η_mean)
        @printf("%-22s %12.6f   (target %.6f)\n", "mean a", imp.a_mean,
                (model.params.μ / (model.params.μ - 1)))
        println("─"^70)
    end

    return (model = model, ws = ws, status = st, growth = gw, φ = φ, implied = imp)
end

# =====================================================================
#  6. Pictures
# =====================================================================

"""
    plot_monopoly(res; save = nothing) -> Plot

Four panels: where the industries sit, what a firm is worth, how hard it
researches, and how likely that research is to pay off.
"""
function plot_monopoly(res; save = nothing)
    model = res.model
    par   = model.params
    a     = nodes(xaxis(model.grid))
    haz   = Hazard(par.η̄, par.θ, par.ε)
    pol   = vec(model.sol.policy.data)
    V     = vec(model.sol.V.data)
    η     = [innovation_prob(pol[i], a[i], haz) for i in eachindex(a)]

    common = (; xlabel = "a = A/w", legend = false, lw = 2, grid = true)

    p1 = plot(a, res.φ; title = "stationary distribution", ylabel = "mass",
              fill = (0, 0.25), common...)
    vline!(p1, [res.implied.a_mean]; ls = :dash, lw = 1, c = :black)
    vline!(p1, [par.μ / (par.μ - 1)]; ls = :dot, lw = 1, c = :red)

    p2 = plot(a, V;   title = "value  V(a)",               ylabel = "V", common...)
    p3 = plot(a, pol; title = "research policy  l(a)",     ylabel = "l", common...)
    p4 = plot(a, η;   title = "innovation probability  η", ylabel = "η",
              ylims = (0, 1), common...)

    ttl = @sprintf("n=1  γ=%.3f  θ=%.2f  ε=%.2f  η̄=%.2f  μ=%.2f  |  g_w=%.5f  ∫p^{1-μ}=%.4f",
                   par.γ, par.θ, par.ε, par.η̄, par.μ,
                   model.sol.aggs.g_w, res.implied.level)
    plt = plot(p1, p2, p3, p4; layout = (2, 2), size = (1100, 720),
               plot_title = ttl, plot_titlefontsize = 9)

    save === nothing || savefig(plt, save)
    return plt
end

end # module
