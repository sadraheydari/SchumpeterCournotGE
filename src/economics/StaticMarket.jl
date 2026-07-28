"""
    StaticMarket

A module for solving the static Cournot competition game with endogenous firm participation 
and perfectly substitutable varieties.

This module computes the closed-form solutions for a given industry state, resolving the 
fixed-point problem of endogenous entry and exit. Active firms are determined by their 
ability to cover marginal costs given the equilibrium industry price.

# Exports
- **Industry Aggregates:** `active_firm_count`, `industry_agg_productivity`, `industry_markup`, `industry_price`
- **Firm Outcomes:** `market_share`, `market_share_with_meta`, `market_revenue`, `market_revenue_with_meta`
"""
module StaticMarket

export active_firm_count, industry_agg_productivity, industry_markup, industry_price,
       market_share, market_share_with_meta,
       market_revenue, market_revenue_with_meta

# ==========================================
# Internal Core Helper
# ==========================================

"""
    _industry_equilibrium(a::AbstractVector{Float64}, μ::Float64)

Internal helper function that resolves the endogenous participation fixed-point problem.

Sorts firms by productivity in descending order and iteratively tests the participation 
constraint. A firm is active if and only if its productivity satisfies:
\$a_{(k)} > \\frac{\\mu k - 1}{\\mu k} \\tilde{a}_k\$

# Arguments
- `a::AbstractVector{Float64}`: Vector of relative productivities for all firms in the industry.
- `μ::Float64`: Elasticity of substitution across goods (\$μ > 1\$).

# Returns
- `NamedTuple{(:n_active, :a_tilde), Tuple{Int, Float64}}`: Contains the equilibrium number 
  of active firms (\$\\tilde{n}\$) and the aggregate industry productivity (\$\\tilde{a}\$).
"""
function _industry_equilibrium(a::AbstractVector{Float64}, μ::Float64)
    a_sorted = sort(a, rev=true)
    n = length(a)
    
    k_star = 1
    a_tilde = a_sorted[1]
    sum_inv_a = 1.0 / a_sorted[1]
    
    for k in 2:n
        current_sum_inv_a = sum_inv_a + 1.0 / a_sorted[k]
        current_a_tilde = k / current_sum_inv_a
        threshold = (μ * k - 1.0) / (μ * k) * current_a_tilde
        
        if a_sorted[k] > threshold
            k_star = k
            a_tilde = current_a_tilde
            sum_inv_a = current_sum_inv_a
        else
            break
        end
    end
    
    return (n_active = k_star, a_tilde = a_tilde)
end

# ==========================================
# Industry-Level Functions
# ==========================================

"""
    active_firm_count(a::AbstractVector{Float64}, μ::Float64) -> Int

Computes the equilibrium number of active participating firms (\$\\tilde{n}\$) in the industry.
"""
function active_firm_count(a::AbstractVector{Float64}, μ::Float64)
    return _industry_equilibrium(a, μ).n_active
end

"""
    industry_agg_productivity(a::AbstractVector{Float64}, μ::Float64) -> Float64

Computes the aggregate productivity (\$\\tilde{a}\$) of the industry, defined as the harmonic mean 
of the productivities of the active firms.
"""
function industry_agg_productivity(a::AbstractVector{Float64}, μ::Float64)
    return _industry_equilibrium(a, μ).a_tilde
end

"""
    industry_markup(n_active::Int, μ::Float64) -> Float64
    industry_markup(a::AbstractVector{Float64}, μ::Float64) -> Float64

Computes the industry aggregate markup (\$m\$), calculated as:
\$m = \\frac{\\mu \\tilde{n}}{\\mu \\tilde{n} - 1}\$

Can be called directly with the number of active firms (`n_active`) to bypass recalculation, 
or with the state vector `a` to compute endogenously.
"""
function industry_markup(n_active::Int, μ::Float64)
    return (μ * n_active) / (μ * n_active - 1.0)
end

function industry_markup(a::AbstractVector{Float64}, μ::Float64)
    n_active = active_firm_count(a, μ)
    return industry_markup(n_active, μ)
end

"""
    industry_price(a_tilde::Float64, m::Float64) -> Float64
    industry_price(a::AbstractVector{Float64}, μ::Float64) -> Float64

Computes the equilibrium industry price (\$p\$), formulated as a markup over the 
aggregate real marginal cost:
\$p = \\frac{m}{\\tilde{a}}\$

Can be called directly with precomputed aggregate productivity (`a_tilde`) and markup (`m`), 
or with the state vector `a` to compute endogenously.
"""
function industry_price(a_tilde::Float64, m::Float64)
    return m / a_tilde
end

function industry_price(a::AbstractVector{Float64}, μ::Float64)
    eq = _industry_equilibrium(a, μ)
    m = industry_markup(eq.n_active, μ)
    return industry_price(eq.a_tilde, m)
end

# ==========================================
# Firm-Level Functions
# ==========================================

"""
    market_share(a::AbstractVector{Float64}, μ::Float64) -> Float64

Computes the focal firm's (represented by `a[1]`) market share (\$s_{own}\$). 
If the firm is below the participation threshold, the market share is strictly `0.0`.

The analytical expression for an active firm is:
\$s_{own} = \\mu \\left(1 - \\frac{1}{m} \\frac{\\tilde{a}}{a_{own}}\\right)\$
"""
function market_share(a::AbstractVector{Float64}, μ::Float64)
    eq = _industry_equilibrium(a, μ)
    m = industry_markup(eq.n_active, μ)
    
    return max(0.0, μ * (1.0 - eq.a_tilde / (m * a[1])))
end

"""
    market_share_with_meta(a::AbstractVector{Float64}, μ::Float64) -> NamedTuple

Computes the focal firm's market share alongside all intermediate industry aggregates. 
Use this function in dynamic programming loops when aggregate state variables are needed 
simultaneously, maintaining type stability while avoiding redundant fixed-point calculations.

# Returns
- `NamedTuple`: `(s_own, n_active, a_tilde, m, p)`
"""
function market_share_with_meta(a::AbstractVector{Float64}, μ::Float64)
    eq = _industry_equilibrium(a, μ)
    m = industry_markup(eq.n_active, μ)
    p = industry_price(eq.a_tilde, m)
    
    s_own = max(0.0, μ * (1.0 - eq.a_tilde / (m * a[1])))
    
    return (s_own = s_own, n_active = eq.n_active, a_tilde = eq.a_tilde, m = m, p = p)
end

"""
    market_revenue(a::AbstractVector{Float64}, y_hat::Float64, μ::Float64) -> Float64

Computes the operational revenue (dividend prior to research costs) for the focal firm (`a[1]`).

Revenue is determined by evaluating the firm's Lerner index (\$\\ell_{own}\$) and applying the 
aggregate demand shifter (\$\\hat{y}\$):
\$\\pi_{own} = p^{1-\\mu} \\ell_{own} \\hat{y}\$

# Arguments
- `a::AbstractVector{Float64}`: Vector of relative productivities, where `a[1]` is the focal firm.
- `y_hat::Float64`: Aggregate demand/output shifter.
- `μ::Float64`: Elasticity of substitution.
"""
function market_revenue(a::AbstractVector{Float64}, y_hat::Float64, μ::Float64)
    eq = _industry_equilibrium(a, μ)
    m = industry_markup(eq.n_active, μ)
    p = industry_price(eq.a_tilde, m)
    a_own = a[1]
    
    s_own = max(0.0, μ * (1.0 - eq.a_tilde / (m * a_own)))
    
    lerner_own = s_own > 0.0 ? (p - 1.0 / a_own) / p * s_own : 0.0
    
    return (p^(1.0 - μ)) * lerner_own * y_hat
end

"""
    market_revenue_with_meta(a::AbstractVector{Float64}, y_hat::Float64, μ::Float64) -> NamedTuple

Computes the focal firm's operational revenue alongside all intermediate industry aggregates. 
Optimized for zero-allocation use inside Dynamic Programming value function iterations.

# Returns
- `NamedTuple`: `(revenue_own, s_own, n_active, a_tilde, m, p)`
"""
function market_revenue_with_meta(a::AbstractVector{Float64}, y_hat::Float64, μ::Float64)
    eq = _industry_equilibrium(a, μ)
    m = industry_markup(eq.n_active, μ)
    p = industry_price(eq.a_tilde, m)
    a_own = a[1]
    
    s_own = max(0.0, μ * (1.0 - eq.a_tilde / (m * a_own)))
    
    lerner_own = s_own > 0.0 ? (p - 1.0 / a_own) / p * s_own : 0.0
    revenue_own = (p^(1.0 - μ)) * lerner_own * y_hat
    
    return (revenue_own = revenue_own, s_own = s_own, n_active = eq.n_active, a_tilde = eq.a_tilde, m = m, p = p)
end

end # module