"""
    SymPolicyEquilibrium

The middle loop. Rivals' research is not exogenous — every firm is choosing
it, and each firm's best response depends on what the others do. This
solves that fixed point by iterating on the belief `policy_comp`:

 1. take `policy_comp` as given and solve the firm's problem
    (`solve_vfi!`), yielding the best response `policy`;
 2. measure how far the best response is from the belief that produced it;
 3. move the belief part of the way toward the response,
    `policy_comp ← λ·policy + (1-λ)·policy_comp`;
 4. repeat until the gap closes.

At a fixed point `policy == policy_comp`: every firm is playing a best
response to what everyone else is playing, which is the symmetric
equilibrium of the game given the aggregates.

# Why the update is damped

Undamped best-response iteration (`λ = 1`) oscillates in games of strategic
substitutes, which is what research is here: if rivals research harder, the
prize from your own innovation falls, so you research less — and next round
they overshoot back. Damping averages the swings out. `λ ≈ 0.1–0.3` is the
usual range; smaller is slower but steadier.

# Two things that are easy to get wrong

**The gap is measured before the update, not after.** The residual that
matters is `‖policy - policy_comp‖_∞` for the belief that *generated*
`policy`. Measuring the change in `policy_comp` instead would report `λ`
times that, so shrinking `λ` would look like faster convergence while
actually being slower. A fixed point you can reach by turning a dial is not
a fixed point.

**The rival cache must be rebuilt after the belief moves.** Under `:exact`
the workspace stores each state's innovation probability under
`policy_comp`; leave it stale and the value iteration keeps answering the
old question, the gap never closes, and nothing visibly fails.
`refresh_rivals!` handles both modes.

# Scale

The tolerance is an absolute sup-norm on a policy that can be very small —
research of order `1e-7` is common when `η̄` is low. Compare
[`policy_residual`](@ref) with [`policy_scale`](@ref) before trusting a
converged flag: a gap of `1e-8` means nothing if the policy is `1e-7`
everywhere.
"""
module SymPolicyEquilibrium

using ..SymStateArrays
using ..DSICModel
using ..ValueIteration

export solve_symmetric_policy!, policy_residual, policy_scale, update_policy_guess!

# =====================================================================
#  Measuring and updating
# =====================================================================

"""
    policy_residual(sol::Solution) -> Float64

`‖policy - policy_comp‖_∞`: how far the firm's best response is from the
belief about its rivals that produced it. Zero at a symmetric equilibrium.

Written as a loop rather than `maximum(abs, A .- B)` to avoid allocating a
temporary the size of the whole state space on every game iteration.
"""
function policy_residual(sol::Solution)
    A, B = sol.policy.data, sol.policy_comp.data
    gap = 0.0
    @inbounds for i in eachindex(A, B)
        d = abs(A[i] - B[i])
        d > gap && (gap = d)
    end
    return gap
end

"""
    policy_scale(sol::Solution) -> Float64

`‖policy‖_∞`, for judging whether a gap is small in any meaningful sense.
A converged absolute gap that is not small *relative to this* has not
converged to anything.
"""
function policy_scale(sol::Solution)
    A = sol.policy.data
    s = 0.0
    @inbounds for i in eachindex(A)
        a = abs(A[i])
        a > s && (s = a)
    end
    return s
end

"""
    update_policy_guess!(sol::Solution, λ) -> Solution

`policy_comp ← λ·policy + (1-λ)·policy_comp`, floored at zero.

The floor is belt-and-braces: `optimal_research` never returns a negative
effort and a convex combination of non-negatives is non-negative, so it
only matters if the policy ever gains another source.
"""
function update_policy_guess!(sol::Solution, λ::Real)
    0 < λ <= 1 || throw(ArgumentError("λ must lie in (0,1] (got $λ)"))
    A, B = sol.policy.data, sol.policy_comp.data
    @inbounds for i in eachindex(A, B)
        B[i] = max(0.0, λ * A[i] + (1 - λ) * B[i])
    end
    return sol
end

# =====================================================================
#  The loop
# =====================================================================

"""
    solve_symmetric_policy!(model::DSIC, ws::VFIWorkspace; kwargs...) -> LoopStatus

Iterate the rival-policy belief to a symmetric equilibrium, writing the
result into `model.sol` and the outcome into `model.sol.sym_policy`.

Keywords:

  * `λ = model.settings.λ_sym_policy` — damping on the belief update
  * `require_inner = true` — a game iteration whose value iteration did not
    converge is chasing noise, so the loop refuses to report convergence
    while that is happening
  * `on_iter(iteration, gap)` — called after each game iteration
  * `on_vfi(iteration, residual)` — passed through to the value iteration,
    for a nested progress bar

Neither the value function nor the belief is reset. Consecutive problems
differ only slightly, so warm starting is most of why this is affordable:
the first value iteration may take hundreds of sweeps and later ones a
handful.

On return `sol.policy` holds the best response to the second-to-last
belief, and `sol.policy_comp` the final belief. At convergence they agree
to within the tolerance; **use `sol.policy` downstream**, since that is the
one that actually solves a firm's problem.
"""
function solve_symmetric_policy!(model::DSIC, ws::VFIWorkspace;
                     λ::Real = model.settings.λ_sym_policy,
                     require_inner::Bool = true,
                     on_iter = nothing,
                     on_vfi = nothing)
    set, sol = model.settings, model.sol

    sol.sym_policy.converged = false
    sol.sym_policy.iters     = 0
    sol.sym_policy.residual  = Inf

    inner_failures = 0
    rising         = 0                  # consecutive increases in the gap
    prev_gap       = Inf

    for iter in 1:set.maxiter_sym_policy
        # --- 1. best response to the current belief -------------------
        solve_vfi!(model, ws; on_iter = on_vfi)
        sol.vfi.converged || (inner_failures += 1)

        # --- 2. how far the response is from the belief ---------------
        #        measured BEFORE the update; see the module docstring
        gap = policy_residual(sol)

        # --- 3. move the belief part of the way toward the response ---
        update_policy_guess!(sol, λ)

        # --- 4. the cached rival probabilities are now stale ----------
        refresh_rivals!(ws, model)

        sol.sym_policy.iters    = iter
        sol.sym_policy.residual = gap
        on_iter === nothing || on_iter(iter, gap)

        # a gap that keeps growing means λ is too large for this game
        rising = gap > prev_gap ? rising + 1 : 0
        rising == 5 && @warn(
            "the best-response gap has risen five iterations running — " *
            "λ_sym_policy is probably too large for these parameters",
            λ, gap, iteration = iter)
        prev_gap = gap

        if gap < set.tol_sym_policy
            sol.sym_policy.converged = !(require_inner && !sol.vfi.converged)
            break
        end
    end

    if inner_failures > 0
        @warn("$inner_failures of $(sol.sym_policy.iters) value iterations did not " *
              "converge; the best responses they produced are unreliable, and " *
              "so is the gap measured against them",
              maxiter_vfi = set.maxiter_vfi, tol_vfi = set.tol_vfi)
    end

    return sol.sym_policy
end

"""
    solve_symmetric_policy!(model::DSIC; mode = :auto, kwargs...) -> LoopStatus

Convenience form that builds a fresh `VFIWorkspace`. Building one scans the
whole state space, so keep the workspace across the aggregate loop rather
than calling this repeatedly.
"""
solve_symmetric_policy!(model::DSIC; mode::Symbol = :auto, kwargs...) =
    solve_symmetric_policy!(model, VFIWorkspace(model; mode = mode); kwargs...)

end # module