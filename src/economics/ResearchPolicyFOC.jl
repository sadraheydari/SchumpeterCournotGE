"""
    ResearchPolicyFOC

The firm's research choice: the innovation hazard, and the `l` that
maximises

    obj(l) = base - l + η(l, a)·ΔEV ,     η(l, a) = 1 - exp(-c·l^θ),
             c = η̄·a^{-ε}

Rather than search the objective, this solves its first-order condition,
which is far better conditioned. Near a smooth maximum the objective is
flat, so a direct search locates the argmax only to about `√eps ≈ 1e-8`
however tight the tolerance — and here the argmax *is* the answer, since
`l` is the policy the game loop iterates on. The FOC root is a transversal
zero with `O(1)` slope, so it resolves to ~`1e-14`.

# The transformation

The condition `ΔEV·η'(l) = 1` becomes, in `x = ln l`,

    G(x) = ln(ΔEV·c·θ) + (θ-1)x - c·e^{θx} = 0 .

For `θ < 1` this is strictly decreasing (`G' = (θ-1) - cθe^{θx} < 0`) and
strictly concave (`G'' = -cθ²e^{θx} < 0`), with `G → +∞` as `x → -∞` and
`G → -∞` as `x → +∞`. So the root is unique, and — because Newton on a
decreasing concave function always lands to the right of the root —
iterating from the upper bracket converges monotonically. Bisection
safeguards it anyway.

Working in `x = ln l` also means the tolerance is automatically relative in
`l`, and the bracket cannot produce a negative trial policy.

# Bracket

`l* ≤ ΔEV` rigorously: `η ≤ 1`, so `obj(l) ≤ base + ΔEV - l`, while
`obj(0) = base`. Anything beyond `ΔEV` is worse than doing no research at
all. The lower end is found by geometric expansion, which also detects the
corner `l* = 0` when no interior root exists.

# Scope

`θ ≤ 1` is required. At `θ = 1` the FOC has the closed form
`l* = ln(ΔEV·c)/c`, used directly. For `θ > 1` the hazard is convex near
zero, the objective is no longer concave, and the FOC can have two roots or
none — [`optimal_research`](@ref) throws rather than return a stationary
point that may not be the maximum. The draft assumes `θ ∈ (0,1)`.

The module is deliberately independent of `Params` and `DSIC`: it takes a
[`Hazard`](@ref), which the solver builds once from the parameters.
"""
module ResearchPolicyFOC

export Hazard, hazard_scale, innovation_prob, innovation_prob_deriv,
       research_objective, research_foc, optimal_research

"Productivities are floored here so the `a^{-ε}` term stays finite at a zero grid node."
const AMIN = 1e-12

@noinline _argerr(msg) = throw(ArgumentError(msg))

"""
    Hazard(η̄, θ, ε)

The innovation technology's three parameters, packaged so the research
functions need not know about `Params`. Build one per solve:

```julia
h = Hazard(par.η̄, par.θ, par.ε)
```

Requires all three non-negative, matching `Params`. Note that `θ > 1` is
accepted here — the hazard itself is well defined — but rejected by
[`optimal_research`](@ref), which needs concavity.
"""
struct Hazard
    η̄::Float64
    θ::Float64
    ε::Float64

    function Hazard(η̄::Real, θ::Real, ε::Real)
        η̄ >= 0 || _argerr("η̄ must be non-negative (got $η̄)")
        θ >= 0 || _argerr("θ must be non-negative (got $θ)")
        ε >= 0 || _argerr("ε must be non-negative (got $ε)")
        return new(Float64(η̄), Float64(θ), Float64(ε))
    end
end

Base.show(io::IO, h::Hazard) =
    print(io, "Hazard(η̄=", h.η̄, ", θ=", h.θ, ", ε=", h.ε, ")")

"""
    hazard_scale(a, h::Hazard) -> Float64

`c = η̄·a^{-ε}`, the state-dependent scale of the innovation hazard. Falling
in `a` when `ε > 0`: a firm that has already climbed finds the next step
harder.
"""
@inline hazard_scale(a::Real, h::Hazard) = h.η̄ * max(a, AMIN)^(-h.ε)

"""
    innovation_prob(l, a, h::Hazard) -> Float64

The draft's Poisson hazard `η(l, a) = 1 - exp(-η̄·l^θ·a^{-ε})`, with the
firm's de-trended productivity standing in for its relative position.

`l` is floored at zero so a stray negative value — from extrapolating a
policy, say — cannot produce a complex power. The result lies in `[0, 1)`
for every finite input.
"""
@inline function innovation_prob(l::Real, a::Real, h::Hazard)
    return 1.0 - exp(-hazard_scale(a, h) * max(l, 0.0)^h.θ)
end

"""
    innovation_prob_deriv(l, a, h::Hazard) -> Float64

`∂η/∂l = c·θ·l^{θ-1}·exp(-c·l^θ)`.

At `l = 0` this is `Inf` for `θ < 1` — which is why the optimum is interior
whenever `ΔEV > 0` — `c` for `θ = 1`, and `0` for `θ > 1`.
"""
@inline function innovation_prob_deriv(l::Real, a::Real, h::Hazard)
    c, θ = hazard_scale(a, h), h.θ
    lp = max(l, 0.0)
    if lp == 0.0
        θ < 1 && return c > 0 ? Inf : 0.0
        θ == 1 && return c
        return 0.0
    end
    return c * θ * lp^(θ - 1.0) * exp(-c * lp^θ)
end

"""
    research_objective(l, base, ΔEV, a, h) -> Float64

`base - l + η(l,a)·ΔEV`, where `base` collects everything independent of
`l`. Provided for testing and diagnostics; the solver never evaluates it.
"""
@inline research_objective(l::Real, base::Real, ΔEV::Real, a::Real, h::Hazard) =
    base - l + innovation_prob(l, a, h) * ΔEV

"""
    research_foc(l, ΔEV, a, h) -> Float64

The first-order condition residual `ΔEV·η'(l,a) - 1`. Zero at an interior
optimum, and the quantity to check when verifying a solution.
"""
@inline research_foc(l::Real, ΔEV::Real, a::Real, h::Hazard) =
    ΔEV * innovation_prob_deriv(l, a, h) - 1.0

# =====================================================================
#  The solver
# =====================================================================

# G(x) and G'(x) with x = ln l; see the module docstring.
@inline _G(x::Float64, lnK::Float64, c::Float64, θ::Float64) =
    lnK + (θ - 1.0) * x - c * exp(θ * x)

@inline _dG(x::Float64, c::Float64, θ::Float64) =
    (θ - 1.0) - c * θ * exp(θ * x)

"""
    optimal_research(ΔEV, a, h::Hazard; xtol = 1e-13, maxit = 60) -> Float64

The research effort maximising `base - l + η(l,a)·ΔEV`, by safeguarded
Newton on the log-space first-order condition.

`ΔEV` is the discounted gain from a successful innovation — the caller's
`Δ·(EV⁺ - EV⁻)`. `base` never enters, since it does not affect the argmax.

Returns exactly `0.0` at a corner: when `ΔEV ≤ 0`, when `η̄ = 0` or `θ = 0`
leave research useless, or when `θ = 1` with `ΔEV·c ≤ 1`. Otherwise the
result is interior and satisfies `0 < l ≤ ΔEV`.

`xtol` is relative in `l`, since the iteration runs in `ln l`.

Throws for `θ > 1`, where the objective is not concave — see the module
docstring.

```julia
h = Hazard(1.0, 0.3, 0.1)
l = optimal_research(2.0, 1.5, h)
abs(research_foc(l, 2.0, 1.5, h)) < 1e-12      # true
```
"""
function optimal_research(ΔEV::Real, a::Real, h::Hazard;
                          xtol::Real = 1e-13, maxit::Integer = 60)
    θ = h.θ
    θ <= 1 || _argerr(
        "optimal_research needs θ ≤ 1: for θ > 1 the hazard is convex near " *
        "zero, the objective is not concave, and the first-order condition " *
        "no longer identifies the maximum (got θ = $θ)")

    Δ = Float64(ΔEV)
    (Δ <= 0.0 || θ <= 0.0 || h.η̄ <= 0.0) && return 0.0

    c = hazard_scale(a, h)
    c > 0.0 || return 0.0

    # θ = 1 is exactly solvable: ΔEV·c·e^{-cl} = 1  ⟹  l = ln(ΔEV·c)/c
    if θ == 1.0
        t = Δ * c
        t > 1.0 || return 0.0                    # corner: η'(0) = c is too small
        return log(t) / c
    end

    lnK = log(Δ) + log(c) + log(θ)

    # Upper end: l* ≤ ΔEV, because η ≤ 1 makes anything beyond it worse
    # than doing no research at all.
    x_hi = log(Δ)
    G_hi = _G(x_hi, lnK, c, θ)
    G_hi > 0.0 && return Δ                       # numerically at the bound

    # Lower end by geometric expansion. G → +∞ as x → -∞ for θ < 1, so this
    # terminates; the cap turns a pathological case into a corner rather
    # than a hang.
    step = 1.0
    x_lo = x_hi - step
    G_lo = _G(x_lo, lnK, c, θ)
    nexp = 0
    while G_lo <= 0.0
        nexp += 1
        nexp > 200 && return 0.0
        step *= 2.0
        x_lo = x_hi - step
        G_lo = _G(x_lo, lnK, c, θ)
    end

    # Safeguarded Newton. G is decreasing and concave, so from the upper
    # bracket Newton descends monotonically onto the root; bisection covers
    # the arithmetic edge cases.
    #
    # The bracket test must be non-strict. `xh` is set to `x` at the top of
    # each iteration, so once Newton converges and `xnew == x`, a strict
    # `xnew < xh` would fail and bisect — discarding the answer and jumping
    # to the middle of the bracket.
    xl, xh = x_lo, x_hi                          # G(xl) > 0 > G(xh)
    x = x_hi
    for _ in 1:maxit
        Gx = _G(x, lnK, c, θ)
        if Gx > 0.0
            xl = x
        else
            xh = x
        end

        d = _dG(x, c, θ)
        xnew = (d < 0.0 && isfinite(d)) ? x - Gx / d : 0.5 * (xl + xh)
        (xl <= xnew <= xh) || (xnew = 0.5 * (xl + xh))

        dx = xnew - x
        x  = xnew
        abs(dx) <= xtol * (1.0 + abs(x)) && break
    end

    l = exp(x)
    return l < Δ ? l : Δ
end

end # module