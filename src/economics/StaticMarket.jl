"""
    StaticMarket

A module for solving the static Cournot competition game with endogenous firm participation
and perfectly substitutable varieties.

This module computes the closed-form solutions for a given industry state, resolving the
fixed-point problem of endogenous entry and exit. Active firms are determined by their
ability to cover marginal costs given the equilibrium industry price.

Naming follows the draft: \$\\tilde{n}\$ is the active-firm count, \$\\tilde{a}\$ the
industry aggregate productivity, \$m\$ the industry markup, \$p\$ the industry price,
\$s_i\$ the firm's market share, \$\\ell_i\$ its Lerner coefficient, and

\$\\hat{d}_i = \\underbrace{p^{1-\\mu}\\,\\ell_i\\,\\hat{y}}_{\\text{operational profit}} - l^r_i\$.

# State

`a` is the vector of relative productivities \$a_i = A_i / w\$, with `a[1]` the focal
("own") firm and the remainder its rivals — the layout `StateArray` uses. Relative
marginal cost is therefore \$1/a_i\$, in the same numéraire as the industry price.

Two input forms are supported. Passing an `NTuple` — what `gridpoint` returns — takes an
unrolled, allocation-free path and is the one to use inside a sweep. Passing an
`AbstractVector` sorts a copy and allocates; it is for interactive and testing use.

# Exports
- **Industry Aggregates:** `active_firm_count`, `industry_agg_productivity`, `industry_markup`, `industry_price`, `industry_output_share`
- **Firm Outcomes:** `market_share`, `market_share_with_meta`, `lerner_coefficient`
- **Payoffs:** `operational_profit`, `operational_profit_with_meta`, `operational_profit_weight`
"""
module StaticMarket

export active_firm_count, industry_agg_productivity, industry_markup, industry_price,
       industry_output_share,
       market_share, market_share_with_meta, lerner_coefficient,
       operational_profit, operational_profit_with_meta, operational_profit_weight

"""
    ProductivityState

Either layout accepted for the productivity vector: an `AbstractVector` (convenient,
allocates) or a `Tuple` (allocation-free, for hot loops).
"""
const ProductivityState = Union{AbstractVector{<:Real},Tuple{Vararg{Real}}}

# ==========================================
# Internal Core Helper
# ==========================================

# Allocation-free descending sort on a tuple. The industry has few firms (n < 10), so
# the compiler unrolls this completely and it never touches the heap.
@inline _sortdesc(t::Tuple{})    = ()
@inline _sortdesc(t::Tuple{Any}) = t
@inline _sortdesc(t::Tuple)      = _insertdesc(t[1], _sortdesc(Base.tail(t)))

@inline _insertdesc(v, ::Tuple{}) = (v,)
@inline function _insertdesc(v, t::Tuple)
    v >= t[1] ? (v, t...) : (t[1], _insertdesc(v, Base.tail(t))...)
end

@noinline _argerr(msg) = throw(ArgumentError(msg))

@inline function _validate(a, μ::Real)
    # μ ≤ 1 makes (μk - 1) non-positive and returns a negative price rather than failing
    μ > 1 || _argerr("μ must exceed 1 (got $μ)")
    length(a) >= 1 || _argerr("the industry needs at least one firm")
    for v in a
        v > 0 || _argerr("productivities must be strictly positive (got $v)")
    end
    return nothing
end

"""
    _equilibrium_from_sorted(b, n, μ)

The participation scan, shared by both input forms. `b` is indexable and sorted in
descending productivity.

Walks up the productivity ranking and stops at the first firm that fails

\$a_{(k)} > \\frac{\\mu k - 1}{\\mu k} \\tilde{a}_k\$

which is equivalent to its marginal cost \$1/a_{(k)}\$ reaching the price it would face.
Stopping at the first failure assumes the active set is contiguous from the top, which is
the standard result for this game.

`k = 1` always satisfies the condition — it reduces to \$\\mu > \\mu - 1\$ — so the scan
can never return an empty industry.
"""
@inline function _equilibrium_from_sorted(b, n::Int, μ::Float64)
    k_star    = 1
    sum_inv_a = 1.0 / b[1]
    a_tilde   = float(b[1])

    @inbounds for k in 2:n
        current_sum_inv_a = sum_inv_a + 1.0 / b[k]
        current_a_tilde   = k / current_sum_inv_a
        threshold         = (μ * k - 1.0) / (μ * k) * current_a_tilde

        if b[k] > threshold
            k_star    = k
            a_tilde   = current_a_tilde
            sum_inv_a = current_sum_inv_a
        else
            break
        end
    end

    return (n_active = k_star, a_tilde = a_tilde)
end

"""
    _industry_equilibrium(a, μ)

Resolve the endogenous participation fixed-point problem.

# Returns
- `NamedTuple{(:n_active, :a_tilde), Tuple{Int, Float64}}`: the equilibrium number of
  active firms (\$\\tilde{n}\$) and the aggregate industry productivity (\$\\tilde{a}\$),
  the harmonic mean over the active firms.

A firm sitting exactly at the participation threshold is treated as inactive. The
equilibrium is degenerate there — such a firm would take zero share, and including it
leaves the price unchanged — so this affects the reported \$\\tilde{n}\$ and
\$\\tilde{a}\$ but no observable quantity.
"""
function _industry_equilibrium(a::AbstractVector{<:Real}, μ::Real)
    _validate(a, μ)
    b = sort!(collect(Float64, a); rev = true)      # allocates; see the module docstring
    return _equilibrium_from_sorted(b, length(b), Float64(μ))
end

@inline function _industry_equilibrium(a::Tuple{Vararg{Real,N}}, μ::Real) where {N}
    _validate(a, μ)
    b = _sortdesc(ntuple(i -> Float64(@inbounds a[i]), Val(N)))
    return _equilibrium_from_sorted(b, N, Float64(μ))
end

# ==========================================
# Industry-Level Functions
# ==========================================

"""
    active_firm_count(a::ProductivityState, μ::Real) -> Int

Computes the equilibrium number of active participating firms (\$\\tilde{n}\$) in the industry.
"""
active_firm_count(a::ProductivityState, μ::Real) = _industry_equilibrium(a, μ).n_active

"""
    industry_agg_productivity(a::ProductivityState, μ::Real) -> Float64

Computes the aggregate productivity (\$\\tilde{a}\$) of the industry, defined as the harmonic mean
of the productivities of the active firms.
"""
industry_agg_productivity(a::ProductivityState, μ::Real) = _industry_equilibrium(a, μ).a_tilde

"""
    industry_markup(n_active::Int, μ::Real) -> Float64
    industry_markup(a::ProductivityState, μ::Real) -> Float64

Computes the industry aggregate markup (\$m\$), calculated as:
\$m = \\frac{\\mu \\tilde{n}}{\\mu \\tilde{n} - 1}\$

Can be called directly with the number of active firms (`n_active`) to bypass recalculation,
or with the state vector `a` to compute endogenously.
"""
industry_markup(n_active::Int, μ::Real) = (μ * n_active) / (μ * n_active - 1.0)

industry_markup(a::ProductivityState, μ::Real) =
    industry_markup(active_firm_count(a, μ), μ)

"""
    industry_price(a_tilde::Real, m::Real) -> Float64
    industry_price(a::ProductivityState, μ::Real) -> Float64

Computes the equilibrium industry price (\$p\$), formulated as a markup over the
aggregate real marginal cost:
\$p = \\frac{m}{\\tilde{a}}\$

Can be called directly with precomputed aggregate productivity (`a_tilde`) and markup (`m`),
or with the state vector `a` to compute endogenously.
"""
industry_price(a_tilde::Real, m::Real) = m / a_tilde

function industry_price(a::ProductivityState, μ::Real)
    eq = _industry_equilibrium(a, μ)
    m  = industry_markup(eq.n_active, μ)
    return industry_price(eq.a_tilde, m)
end

"""
    industry_output_share(p::Real, μ::Real) -> Float64
    industry_output_share(a::ProductivityState, μ::Real) -> Float64

The industry's share of total output, \$s(j) = p(j)^{1-\\mu}\$ — the factor that turns a
firm's Lerner coefficient into its profit as a fraction of aggregate output. The
economy-wide Lerner index \$\\mathscr{L}\$ is the output-share-weighted average of
industry Lerner coefficients, so this is the weight it uses.
"""
industry_output_share(p::Real, μ::Real) = p^(1.0 - μ)

industry_output_share(a::ProductivityState, μ::Real) =
    industry_output_share(industry_price(a, μ), μ)

# ==========================================
# Firm-Level Functions
# ==========================================

@inline _own_share(a_own::Real, a_tilde::Float64, m::Float64, μ::Real) =
    max(0.0, μ * (1.0 - a_tilde / (m * a_own)))

# ℓ = ((p - 1/a)/p)·s. A shut-out firm has s = 0 and a negative price-cost margin, so the
# branch keeps ℓ at exactly zero rather than -0.0.
@inline _own_lerner(a_own::Real, p::Float64, s_own::Float64) =
    s_own > 0.0 ? (p - 1.0 / a_own) / p * s_own : 0.0

"""
    market_share(a::ProductivityState, μ::Real) -> Float64

Computes the focal firm's (represented by `a[1]`) market share (\$s_{own}\$).
If the firm is below the participation threshold, the market share is strictly `0.0`.

The analytical expression for an active firm is:
\$s_{own} = \\mu \\left(1 - \\frac{1}{m} \\frac{\\tilde{a}}{a_{own}}\\right)\$
"""
function market_share(a::ProductivityState, μ::Real)
    eq = _industry_equilibrium(a, μ)
    m  = industry_markup(eq.n_active, μ)
    return _own_share(a[1], eq.a_tilde, m, μ)
end

"""
    market_share_with_meta(a::ProductivityState, μ::Real) -> NamedTuple

Computes the focal firm's market share alongside all intermediate industry aggregates.
Use this function in dynamic programming loops when aggregate state variables are needed
simultaneously, maintaining type stability while avoiding redundant fixed-point calculations.

# Returns
- `NamedTuple`: `(s_own, n_active, a_tilde, m, p)`
"""
function market_share_with_meta(a::ProductivityState, μ::Real)
    eq = _industry_equilibrium(a, μ)
    m  = industry_markup(eq.n_active, μ)
    p  = industry_price(eq.a_tilde, m)
    s_own = _own_share(a[1], eq.a_tilde, m, μ)
    return (s_own = s_own, n_active = eq.n_active, a_tilde = eq.a_tilde, m = m, p = p)
end

"""
    lerner_coefficient(a::ProductivityState, μ::Real) -> Float64

The focal firm's Lerner coefficient \$\\ell_{own}\$, the draft's price–cost margin scaled
by market share:

\$\\ell_i = \\frac{p - 1/a_i}{p}\\, s_i\$

Zero for a shut-out firm. This is the \$\\ell\$ that appears in the dividend, and the
object that aggregates — weighted by [`industry_output_share`](@ref) — into the
economy-wide Lerner index \$\\mathscr{L}\$.
"""
function lerner_coefficient(a::ProductivityState, μ::Real)
    eq = _industry_equilibrium(a, μ)
    m  = industry_markup(eq.n_active, μ)
    p  = industry_price(eq.a_tilde, m)
    a_own = a[1]
    return _own_lerner(a_own, p, _own_share(a_own, eq.a_tilde, m, μ))
end

# ==========================================
# Payoffs
# ==========================================

"""
    operational_profit_weight(a::ProductivityState, μ::Real) -> Float64

The state-only coefficient \$p^{1-\\mu}\\,\\ell_{own}\$, so that

\$\\pi_{own} = \\texttt{operational\\_profit\\_weight}(a, \\mu) \\cdot \\hat{y}\$.

It depends on nothing that the solver iterates — not \$\\hat{y}\$, not the policies, not
the aggregates — so it can be computed once per state into a `StateArray` and reused
across every sweep of every loop.
"""
function operational_profit_weight(a::ProductivityState, μ::Real)
    eq = _industry_equilibrium(a, μ)
    m  = industry_markup(eq.n_active, μ)
    p  = industry_price(eq.a_tilde, m)
    a_own = a[1]
    s_own = _own_share(a_own, eq.a_tilde, m, μ)
    return industry_output_share(p, μ) * _own_lerner(a_own, p, s_own)
end

"""
    operational_profit(a::ProductivityState, y_hat::Real, μ::Real) -> Float64

The focal firm's operational profit — the draft's

\$\\pi_{own} = p^{1-\\mu}\\,\\ell_{own}\\,\\hat{y}\$,

i.e. the dividend *before* research spending is netted out: \$\\hat{d}_i = \\pi_i - l^r_i\$.

Equal to `operational_profit_weight(a, μ) * y_hat`; prefer that form when \$\\hat{y}\$
varies but the state does not.

# Arguments
- `a::ProductivityState`: Vector of relative productivities, where `a[1]` is the focal firm.
- `y_hat::Real`: Aggregate demand/output shifter \$\\hat{y}\$.
- `μ::Real`: Elasticity of substitution.
"""
operational_profit(a::ProductivityState, y_hat::Real, μ::Real) =
    operational_profit_weight(a, μ) * y_hat

"""
    operational_profit_with_meta(a::ProductivityState, y_hat::Real, μ::Real) -> NamedTuple

Computes the focal firm's operational profit alongside all intermediate industry aggregates.
Type-stable and, when `a` is a `Tuple`, allocation-free — the form to use inside a value
function iteration. Passing an `AbstractVector` sorts a copy and therefore allocates.

# Returns
- `NamedTuple`: `(profit_own, ℓ_own, s_own, n_active, a_tilde, m, p)`
"""
function operational_profit_with_meta(a::ProductivityState, y_hat::Real, μ::Real)
    eq = _industry_equilibrium(a, μ)
    m  = industry_markup(eq.n_active, μ)
    p  = industry_price(eq.a_tilde, m)
    a_own = a[1]

    s_own      = _own_share(a_own, eq.a_tilde, m, μ)
    ℓ_own      = _own_lerner(a_own, p, s_own)
    profit_own = industry_output_share(p, μ) * ℓ_own * y_hat

    return (profit_own = profit_own, ℓ_own = ℓ_own, s_own = s_own,
            n_active = eq.n_active, a_tilde = eq.a_tilde, m = m, p = p)
end

end # module