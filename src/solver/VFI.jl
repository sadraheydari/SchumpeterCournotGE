"""
    ValueIteration

The inner loop. Given the aggregates `(g_w, g_y, ŷ)` and the frozen rival
policy `policy_comp`, solve the firm's dynamic problem for `V` and `policy`.

# The equation

With `a = A/w` and the de-trended value `V̂ = V/w`,

    V̂(a) = max_{l ≥ 0} { π(a)·ŷ - l + Δ·[ η(l,a)·EV⁺ + (1-η(l,a))·EV⁻ ] }

    Δ = β·(1+g_y)^{-σ}·(1+g_w)

`EV⁺` and `EV⁻` are next period's expected value with and without a
successful own innovation, averaging over the `2^{n-1}` combinations of
rival successes. Next period's state is `γa/(1+g_w)` where a firm
innovated and `a/(1+g_w)` where it did not — generally off-grid, hence the
interpolation.

# Reading order

The file is arranged in the order the work happens:

 1. `VFIWorkspace` — everything fixed for the whole solve, computed once
 2. `rival_etas` — how likely each rival is to innovate
 3. `continuation_values` — the `EV⁺`, `EV⁻` expectation
 4. `bellman_state` — one state: the three pieces above plus the choice of `l`
 5. `solve_vfi!` — sweep every state until the value stops moving

Steps 2–4 are pure functions of one state. You can call `bellman_state` on
its own in the REPL and check it by hand; the sweep contains no economics.

# What is fixed, and why it matters

Three things do not change while the value function iterates, so all three
are computed once into the workspace:

  * the **state geometry** — which column of the array each rival multiset
    occupies, so the sweep never re-ranks a state;
  * the **operational profit weight** `p^{1-μ}·ℓ`, which depends on the
    state alone — not on `ŷ`, the policies, or any aggregate — so it
    survives all three loops and is multiplied by `ŷ` at the point of use;
  * the **rivals' innovation probabilities**, which change only when the
    game loop updates `policy_comp`.

# Two ways to read a rival's policy

A rival's view of a state swaps the first component with one of the
symmetric ones. Whether that is still a grid state depends on the axes:

  * **`:exact`** — when the `x` and `y` axes are the same grid, the permuted
    state *is* a grid state, so a rival's innovation probability is a plain
    array read;
  * **`:interp`** — otherwise `policy_comp` is interpolated at the permuted
    coordinates and `η` applied to the result. Slower and slightly less
    accurate off-grid, but it allows different resolutions on the two axes.

`VFIWorkspace` picks `:exact` when the axes match. The two agree exactly at
grid points, since interpolating at a node returns the node's value.

The interpolated path interpolates the *policy* and applies `η` afterwards,
rather than interpolating `η` itself: a policy floored at zero always gives
`η ∈ [0,1)`, whereas extrapolating `η` could leave the unit interval and
produce a negative probability.
"""
module ValueIteration

using ..SymStateArrays
using ..StateGrids
using ..DSICModel
using ..StaticMarket
using ..ResearchPolicyFOC

export VFIWorkspace, solve_vfi!, refresh_rivals!, rival_mode,
       bellman_state, rival_etas, continuation_values, contraction_modulus

# =====================================================================
#  1. Workspace — everything computed once
# =====================================================================

"""
    VFIWorkspace(model::DSIC; mode = :auto)

Everything the value iteration needs that the value iteration does not
change. Build it once and keep it: constructing one scans the whole state
space.

`mode` is `:exact`, `:interp`, or `:auto`, which picks `:exact` when the two
grid axes match. See the module docstring.

The `rivals` field holds *either* a `StateArray` of precomputed `η` (exact)
*or* an `Interpolant` over `policy_comp` (interpolated). Which one it is
lives in the workspace's **type**, so the sweep picks the right code path by
dispatch rather than by testing a flag at every state.
"""
struct VFIWorkspace{N,M,A<:StateArray{Float64},R}
    combos::Vector{NTuple{M,Int}}   # the rival multisets, one per column
    cols::Vector{Int}               # which column each multiset occupies
    weight::A                       # p^{1-μ}·ℓ, the profit per unit of ŷ
    rivals::R                       # see above
    hazard::Hazard                  # (η̄, θ, ε), lifted out of Params once
    Vold::A                         # last sweep's value function
    diffs::Vector{Float64}          # per-column sup-norm change
end

"""
    rival_mode(ws) -> Symbol

`:exact` if rivals' policies are read straight off the grid, `:interp` if
they are interpolated.
"""
rival_mode(::VFIWorkspace{N,M,A,<:StateArray}) where {N,M,A}  = :exact
rival_mode(::VFIWorkspace{N,M,A,<:Interpolant}) where {N,M,A} = :interp

function VFIWorkspace(model::DSIC; mode::Symbol = :auto)
    par, grid, sol = model.params, model.grid, model.sol
    N = state_length(par)            # length of a state
    M = N - 1                        # how many rivals

    mode in (:auto, :exact, :interp) ||
        throw(ArgumentError("mode must be :auto, :exact or :interp, got $(repr(mode))"))
    axes_match = nodes(xaxis(grid)) == nodes(yaxis(grid))
    use_exact  = mode === :auto ? axes_match : mode === :exact
    (use_exact && !axes_match) && throw(ArgumentError(
        "mode = :exact needs identical x and y axes: a rival's view swaps " *
        "the first component with a symmetric one, which is only a grid " *
        "state when both axes are the same grid. Use :interp, or set " *
        "kx == ky with matching spacing."))

    # One entry per column of the value array.
    combos = _multisets(length(yaxis(grid)), Val(M))
    cols   = [stateindex(sol.V, (1, c...))[2] for c in combos]

    weight = statearray(grid, N)
    Vold   = statearray(grid, N)
    kx     = length(xaxis(grid))

    for (ci, c) in enumerate(combos), gi in 1:kx
        a = gridpoint(grid, (gi, c...))          # grid indices -> real values
        @inbounds weight.data[gi, cols[ci]] = operational_profit_weight(a, par.μ)
    end

    rivals = use_exact ? statearray(grid, N) : Interpolant(sol.policy_comp, grid)
    hazard = Hazard(par.η̄, par.θ, par.ε)

    ws = VFIWorkspace{N,M,typeof(weight),typeof(rivals)}(
        combos, cols, weight, rivals, hazard, Vold,
        Vector{Float64}(undef, length(combos)))

    refresh_rivals!(ws, model)
    return ws
end

"""
    _multisets(k, Val(M)) -> Vector{NTuple{M,Int}}

Every non-decreasing length-`M` tuple drawn from `1:k` — one per column of a
`StateArray`, since the rivals are exchangeable and so stored sorted.

`Val(M)` passes the length at the *type* level, which is what lets `ntuple`
build a fixed-size tuple the compiler can unroll. A plain `M::Int` would
give a tuple whose length is unknown until runtime, and the result would be
type-unstable.
"""
function _multisets(k::Int, ::Val{M}) where {M}
    M == 0 && return [()]                        # n = 1: no rivals at all
    out = NTuple{M,Int}[]
    y = ntuple(_ -> 1, Val(M))
    while true
        push!(out, ntuple(i -> y[i], Val(M)))
        i = M
        while i >= 1 && y[i] == k                # find the rightmost index
            i -= 1                               # that can still be raised
        end
        i == 0 && break
        y[i] += 1
        for j in (i + 1):M                       # reset the tail to keep it
            y[j] = y[i]                          # non-decreasing
        end
    end
    return out
end

"""
    refresh_rivals!(ws, model::DSIC)

Bring the rival-policy cache up to date after the game loop damps
`policy_comp`. Not needed between value iterations, where that policy is
frozen.

Under `:exact` this recomputes every state's `η`. Under `:interp` it does
nothing: the `Interpolant` holds `policy_comp` by reference, so it already
sees the update.
"""
function refresh_rivals!(ws::VFIWorkspace{N,M,A,<:StateArray},
                         model::DSIC) where {N,M,A}
    grid, sol = model.grid, model.sol
    kx = length(xaxis(grid))
    for (ci, c) in enumerate(ws.combos), gi in 1:kx
        col   = ws.cols[ci]
        a_own = xaxis(grid)[gi]
        l     = max(sol.policy_comp.data[gi, col], 0.0)
        @inbounds ws.rivals.data[gi, col] = innovation_prob(l, a_own, ws.hazard)
    end
    return ws
end

refresh_rivals!(ws::VFIWorkspace{N,M,A,<:Interpolant}, ::DSIC) where {N,M,A} = ws

# =====================================================================
#  2. Rivals' innovation probabilities
# =====================================================================

"""
    _rival_view(gi, y, m, Val(N)) -> NTuple{N,Int}

The state's grid *indices* as rival `m` sees them: that rival moves to the
front, the current focal firm joins the tail, the other rivals follow.

    (gi, y₁, y₂, y₃)  seen by rival 2  ->  (y₂, gi, y₁, y₃)
"""
@inline function _rival_view(gi::Int, y::NTuple{M,Int}, m::Int,
                             ::Val{N}) where {M,N}
    return ntuple(Val(N)) do k
        if k == 1
            @inbounds y[m]                       # the rival becomes focal
        elseif k == 2
            gi                                   # we become one of its rivals
        else
            j = k - 2                            # the remaining rivals,
            @inbounds j < m ? y[j] : y[j + 1]    # skipping index m
        end
    end
end

"""
    _rival_point(a, m, Val(N)) -> NTuple{N,Float64}

The same permutation in real coordinates, for the interpolated path where
the permuted state need not be a grid state.
"""
@inline function _rival_point(a::NTuple{N,Float64}, m::Int, ::Val{N}) where {N}
    return ntuple(Val(N)) do k
        if k == 1
            @inbounds a[m + 1]
        elseif k == 2
            @inbounds a[1]
        else
            j = k - 2
            @inbounds j < m ? a[j + 1] : a[j + 2]
        end
    end
end

"""
    rival_etas(ws, gi, y, a) -> NTuple{M,Float64}

Each rival's probability of innovating this period under the frozen
`policy_comp`.

Two methods, chosen by the type of `ws.rivals` — no runtime branch.
"""
@inline function rival_etas(ws::VFIWorkspace{N,M,A,<:StateArray}, gi::Int,
                            y::NTuple{M,Int}, a::NTuple{N,Float64}) where {N,M,A}
    # exact: the permuted state is a grid state, so this is an array read
    return ntuple(m -> @inbounds(ws.rivals[_rival_view(gi, y, m, Val(N))]), Val(M))
end

@inline function rival_etas(ws::VFIWorkspace{N,M,A,<:Interpolant}, gi::Int,
                            y::NTuple{M,Int}, a::NTuple{N,Float64}) where {N,M,A}
    # interpolated: read the policy off the grid, then apply the hazard
    return ntuple(Val(M)) do m
        l = ws.rivals(_rival_point(a, m, Val(N)))
        @inbounds innovation_prob(l, a[m + 1], ws.hazard)
    end
end

# =====================================================================
#  3. The continuation values
# =====================================================================

"""
    continuation_values(γ, Ṽ, a, η_riv, scale) -> (EV_succ, EV_fail)

Next period's expected value, with and without a successful own innovation.

Each of the `2^M` bit patterns says which rivals innovated. The rivals'
outcomes do not depend on the firm's own `l`, so this is computed once per
state and reused for every trial `l` in the maximisation — which is what
makes the choice of `l` cheap relative to this step.

`scale = 1/(1+g_w)` de-trends next period's state; a firm that innovates
multiplies its productivity by `γ` first.
"""
@inline function continuation_values(γ::Float64, Ṽ, a::NTuple{N,Float64},
                                     η_riv::NTuple{M,Float64},
                                     scale::Float64) where {N,M}
    EV_succ = 0.0
    EV_fail = 0.0

    @inbounds for mask in 0:((1 << M) - 1)
        # probability that exactly the rivals flagged in `mask` innovate
        prob = 1.0
        for m in 1:M
            innovated = ((mask >> (m - 1)) & 1) == 1
            prob *= innovated ? η_riv[m] : (1.0 - η_riv[m])
        end
        prob == 0.0 && continue

        # next period's rival productivities under this pattern
        next_rivals = ntuple(Val(M)) do m
            innovated = ((mask >> (m - 1)) & 1) == 1
            (innovated ? γ * a[m + 1] : a[m + 1]) * scale
        end

        EV_succ += prob * Ṽ((γ * a[1] * scale, next_rivals...))
        EV_fail += prob * Ṽ((a[1] * scale, next_rivals...))
    end

    return (EV_succ, EV_fail)
end

# =====================================================================
#  4. One state
# =====================================================================

"""
    bellman_state(par, ws, Ṽ, grid, disc, ŷ, g_w, gi, y, col) -> (v, l)

The right-hand side of the Bellman equation at a single state: the new value
and the research effort that attains it.

A pure function — nothing is mutated. Call it directly to check a state by
hand, or to verify a first-order condition at a solution.

The choice of `l` is delegated to `optimal_research`, which solves the
first-order condition rather than searching the objective. `ΔEV ≤ 0` is
returned as exactly zero research, since a firm gains nothing from a
success it does not value.
"""
function bellman_state(par::Params, ws::VFIWorkspace{N,M}, Ṽ, grid,
                       disc::Float64, ŷ::Float64, g_w::Float64,
                       gi::Int, y::NTuple{M,Int}, col::Int) where {N,M}
    a     = gridpoint(grid, (gi, y...))          # grid indices -> real values
    a_own = @inbounds a[1]
    scale = 1.0 / (1.0 + g_w)

    profit = @inbounds ws.weight.data[gi, col] * ŷ
    η_riv  = rival_etas(ws, gi, y, a)
    EV_succ, EV_fail = continuation_values(par.γ, Ṽ, a, η_riv, scale)

    ΔEV  = disc * (EV_succ - EV_fail)            # the prize from innovating
    base = profit + disc * EV_fail               # the value of no research

    l = optimal_research(ΔEV, a_own, ws.hazard)
    l == 0.0 && return (base, 0.0)

    v = base - l + innovation_prob(l, a_own, ws.hazard) * ΔEV
    return v >= base ? (v, l) : (base, 0.0)      # never worse than not trying
end

# =====================================================================
#  5. The sweep
# =====================================================================

"""
    contraction_modulus(par::Params, aggs::Aggregates) -> Float64

    Λ = β·(1+g_y)^{-σ}·γ

The factor by which one round of "innovate and carry the value forward"
scales the value function. The Bellman operator is a contraction only if
`Λ < 1`; at `Λ ≥ 1` the firm's problem has **no bounded solution** and the
iteration diverges rather than failing.

Why this and not something involving `μ` or `g_w`: above the top grid point
the value is *extrapolated linearly*, so its tail grows linearly in `a`
whatever curvature the profit function has. A firm that innovates moves to
`γa/(1+g_w)`, worth `Δ·γ/(1+g_w)` times as much, and
`Δ = β(1+g_y)^{-σ}(1+g_w)` cancels the `(1+g_w)`.

Read economically it is the transversality condition: the discounted
productivity step must be below one. Setting `γ > 1` while holding
`g_y = 0` violates it — if firms innovate, output grows, and it is that
growth in the discount factor that keeps the problem bounded.
"""
contraction_modulus(par::Params, aggs::Aggregates) =
    par.β * (1.0 + aggs.g_y)^(-par.σ) * par.γ

"""
    solve_vfi!(model::DSIC, ws::VFIWorkspace; on_iter = nothing) -> LoopStatus

Apply the Bellman operator until the value function stops moving, writing
`V` and `policy` into `model.sol` and the outcome into `model.sol.vfi`.

**Jacobi, not Gauss–Seidel.** Each sweep reads last sweep's values from
`ws.Vold` and writes this sweep's into `sol.V`. Gauss–Seidel — reading the
values already updated in this sweep — converges in fewer sweeps, but the
answer would then depend on the order states are visited, which makes
threading non-deterministic. Here each thread owns whole columns, so no two
threads touch the same entry.

`V` is **not** reset. Warm starting from the previous outer iteration is
the reason `Solution` is mutable: consecutive problems are nearly identical,
so a warm start takes a handful of sweeps where a cold one takes hundreds.

`on_iter(iteration, residual)` runs after each sweep — a hook for a progress
bar or a convergence trace.
"""
function solve_vfi!(model::DSIC, ws::VFIWorkspace{N,M};
                    on_iter = nothing) where {N,M}
    par, grid, sol, set = model.params, model.grid, model.sol, model.settings
    aggs = sol.aggs

    # the de-trended discount factor, constant for the whole solve
    disc = par.β * (1.0 + aggs.g_y)^(-par.σ) * (1.0 + aggs.g_w)
    ŷ    = aggs.ŷ
    g_w  = aggs.g_w
    kx   = length(xaxis(grid))

    # Vold is overwritten in place each sweep, so this interpolant — which
    # holds it by reference — stays valid throughout.
    Ṽ = Interpolant(ws.Vold, grid)

    Λ = contraction_modulus(par, aggs)
    Λ < 1.0 || @warn(
        "the Bellman operator is not a contraction (Λ = β(1+g_y)^-σ·γ ≥ 1), " *
        "so the firm's problem has no bounded solution and this will " *
        "diverge; raise g_y, or lower β or γ", Λ, par.β, par.γ,
        g_y = aggs.g_y, σ = par.σ)

    sol.vfi.converged = false
    sol.vfi.iters     = 0
    sol.vfi.residual  = Inf

    for iter in 1:set.maxiter_vfi
        copyto!(ws.Vold.data, sol.V.data)        # this sweep reads Vold

        Threads.@threads for ci in eachindex(ws.combos)
            y    = ws.combos[ci]
            col  = ws.cols[ci]
            worst = 0.0
            @inbounds for gi in 1:kx
                v, l = bellman_state(par, ws, Ṽ, grid, disc, ŷ, g_w, gi, y, col)
                worst = max(worst, abs(v - ws.Vold.data[gi, col]))
                sol.V.data[gi, col]      = v
                sol.policy.data[gi, col] = l
            end
            ws.diffs[ci] = worst
        end

        residual = maximum(ws.diffs)
        sol.vfi.iters    = iter
        sol.vfi.residual = residual

        # Divergence shows up here long before it would as a wrong answer.
        isfinite(residual) || error(
            "value iteration diverged at sweep $iter: the residual is " *
            "$residual. Check contraction_modulus(params, aggregates) = " *
            "$(contraction_modulus(par, aggs)), which must be below 1.")
        on_iter === nothing || on_iter(iter, residual)

        if residual < set.tol_vfi
            sol.vfi.converged = true
            break
        end
    end

    return sol.vfi
end

"""
    solve_vfi!(model::DSIC; mode = :auto, kwargs...) -> LoopStatus

Convenience form that builds a fresh workspace. Building one scans the whole
state space, so keep the workspace across the outer loops rather than
calling this repeatedly.
"""
solve_vfi!(model::DSIC; mode::Symbol = :auto, kwargs...) =
    solve_vfi!(model, VFIWorkspace(model; mode = mode); kwargs...)

end # module