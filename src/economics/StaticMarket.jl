module StaticMarket

export active_firm_count, industry_agg_productivity, industry_markup, industry_price,
       market_share, market_share_with_meta,
       market_revenue, market_revenue_with_meta

# ==========================================
# Internal Core Helper
# ==========================================

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

function active_firm_count(a::AbstractVector{Float64}, μ::Float64)
    return _industry_equilibrium(a, μ).n_active
end

function industry_agg_productivity(a::AbstractVector{Float64}, μ::Float64)
    return _industry_equilibrium(a, μ).a_tilde
end

function industry_markup(n_active::Int, μ::Float64)
    return (μ * n_active) / (μ * n_active - 1.0)
end

function industry_markup(a::AbstractVector{Float64}, μ::Float64)
    n_active = active_firm_count(a, μ)
    return industry_markup(n_active, μ)
end

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

function market_share(a::AbstractVector{Float64}, μ::Float64)
    eq = _industry_equilibrium(a, μ)
    m = industry_markup(eq.n_active, μ)
    
    return max(0.0, μ * (1.0 - eq.a_tilde / (m * a[1])))
end

function market_share_with_meta(a::AbstractVector{Float64}, μ::Float64)
    eq = _industry_equilibrium(a, μ)
    m = industry_markup(eq.n_active, μ)
    p = industry_price(eq.a_tilde, m)
    
    s_own = max(0.0, μ * (1.0 - eq.a_tilde / (m * a[1])))
    
    return (s_own = s_own, n_active = eq.n_active, a_tilde = eq.a_tilde, m = m, p = p)
end

function market_revenue(a::AbstractVector{Float64}, y_hat::Float64, μ::Float64)
    eq = _industry_equilibrium(a, μ)
    m = industry_markup(eq.n_active, μ)
    p = industry_price(eq.a_tilde, m)
    a_own = a[1]
    
    s_own = max(0.0, μ * (1.0 - eq.a_tilde / (m * a_own)))
    
    lerner_own = s_own > 0.0 ? (p - 1.0 / a_own) / p * s_own : 0.0
    
    return (p^(1.0 - μ)) * lerner_own * y_hat
end

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