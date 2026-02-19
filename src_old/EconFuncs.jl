"""
    harmonic_mean(vec::Vector{Float64}) -> Float64

Compute the harmonic mean of a vector of `Float64` values.

The harmonic mean is defined as

    H = n / sum(1 / xᵢ)

where `n` is the length of the vector and `xᵢ` are the elements of `vec`.

# Arguments
- `vec::Vector{Float64}`: A vector of real numbers.

# Returns
- `Float64`: The harmonic mean of the elements in `vec`.

# Notes
- All elements of `vec` must be nonzero.
- The harmonic mean is only defined for vectors whose elements are either all positive or all negative.
- If any element is zero, a `DivideError` will be thrown.
"""
function harmonic_mean(vec:: Vector{Float64}):: Float64
    res = 0.0
    for i in eachindex(vec)
        res += 1.0 / vec[i]
    end
    return length(vec) / res
end;



"""
    find_active_threshold(
        A_vec_sorted::Vector{Float64},
        start_index::Int64,
        end_index::Int64,
        p::ModelParameters
    ) -> Tuple{Int64, Float64}

Compute the equilibrium participation threshold using a bisection procedure.

For a given sorted productivity profile `A_vec_sorted`, the function identifies
the size of the equilibrium active set `ñ` and the associated harmonic mean `Ã`
such that the implied participation threshold is satisfied.

The algorithm implements a recursive bisection search over the index interval
`[start_index, end_index]`. At each step, it checks whether a candidate active
set of size `ñ` satisfies the participation condition and narrows the search
interval accordingly. The procedure exploits the monotonicity induced by the
sorted productivity profile.

The function operationalizes the following equilibrium concept:

**Equilibrium Active Set.**
A set `𝒩ᴾ ⊆ 𝒩` is an equilibrium active set if and only if:

1. **Internal Consistency:** For all `i ∈ 𝒩ᴾ`,
       Aᵢ ≥ ((|𝒩ᴾ| - σ) / |𝒩ᴾ|) · Ā(𝒩ᴾ).

2. **External Consistency:** For all `i ∉ 𝒩ᴾ`,
       Aᵢ < ((|𝒩ᴾ| + 1 - σ) / (|𝒩ᴾ| + 1)) · Ā(𝒩ᴾ ∪ {i}).

Under a sorted productivity profile, the equilibrium active set is characterized
by a cutoff index `ñ`, and the function returns:

- `ñ`: the equilibrium number of active firms,
- `Ã`: the harmonic mean of productivities among the active agents,
         i.e. `Ã = harmonic_mean(A_vec_sorted[1:ñ])`.

# Arguments
- `A_vec_sorted::Vector{Float64}`: Productivity profile sorted in ascending order.
- `start_index::Int64`: Lower bound (inclusive) of the search interval.
- `end_index::Int64`: Upper bound (inclusive) of the search interval.
- `p::ModelParameters`: Model parameters, including:
    - `σ`: risk aversion.

# Returns
- `Tuple{Int64, Float64}`: `(ñ, Ã)` where `ñ` is the equilibrium active set
  size and `Ã` is the corresponding harmonic mean productivity.

# Assumptions
- `A_vec_sorted` is sorted in ascending order.
- `1 ≤ start_index ≤ end_index ≤ length(A_vec_sorted)`.
- All relevant productivity values are positive.
"""
function find_active_threshold(
        A_vec_sorted:: Vector{Float64},
        start_index:: Int64, 
        end_index:: Int64,
        p:: ModelParameters
    ):: Tuple{Int64, Float64}
    
    # Termination condition for the recursion
    if start_index >= end_index
        Ã = harmonic_mean(A_vec_sorted[1:start_index]) 
        return (start_index, Ã)
    end

    # Check the current endpoint
    ñ = end_index
    Ã = harmonic_mean(A_vec_sorted[1:end_index]) 
    threshold = (ñ - p.σ) / ñ * Ã
    if A_vec_sorted[end_index] >= threshold 
        return (end_index, Ã)
    end

    # Assume mid_index is active and check if it satisfies the condition
    mid_index = floor(Int64, (start_index + end_index) / 2)
    if mid_index == start_index 
        Ã = harmonic_mean(A_vec_sorted[1:mid_index])
        return (mid_index, Ã)
    end

    ñ = mid_index
    Ã = harmonic_mean(A_vec_sorted[1:mid_index])
    threshold = (ñ - p.σ) / ñ * Ã

    if A_vec_sorted[mid_index] >= threshold
        return find_active_threshold(A_vec_sorted, mid_index, end_index-1, p)
    else
        return find_active_threshold(A_vec_sorted, start_index, mid_index - 1, p) 
    end
end;


"""
    is_active(A_vec::Vector{Int64}, p::ModelParameters)
        -> Tuple{Vector{Bool}, Float64}

Determine the equilibrium active firms given their positions on the quality ladder.

Each firm `i` is characterized by its ladder position `τᵢ = A_vec[i]`, with
productivity

    Aᵢ = γ^{τᵢ},

where `γ` (contained in `p::ModelParameters`) is the technology growth factor
associated with a successful innovation step on the quality ladder.

The function ranks firms in descending order of productivity, constructs the
implied productivity profile `{γ^{τᵢ}}`, and calls `find_active_threshold`
to compute the equilibrium participation cutoff using a bisection procedure.
The resulting cutoff index determines the size of the equilibrium active set,
and the outcome is mapped back to the original firm ordering.

# Arguments
- `A_vec::Vector{Int64}`: Vector of firms’ positions on the quality ladder,
  where `A_vec[i] = τᵢ`.
- `p::ModelParameters`: Model parameters, including:
    - `γ`: innovation step size (technology growth factor per successful innovation),,
    - `σ`: risk aversion.

# Returns
- `Tuple{Vector{Bool}, Float64}`:
    - `active_set::Vector{Bool}`: Boolean vector where `active_set[i] = true`
      if firm `i` is active in equilibrium.
    - `Ã::Float64`: Harmonic mean productivity among active firms.
"""
function is_active(A_vec:: Vector{Int64}, p:: ModelParameters):: Tuple{Vector{Bool}, Float64} 
    
    sorted_indices = sortperm(A_vec, rev=true)
    rank_sorted = invperm(sorted_indices)
    A_vec_sorted = [(p.γ ^ (A_vec[i] - 1)) for i in sorted_indices]

    last_active_index, Ã = find_active_threshold(A_vec_sorted, 1, length(A_vec), p)
    active_set =  [(i <= last_active_index) for i in rank_sorted]
    return active_set, Ã
end;



"""
    market_share(A_vec::Vector{Int64}, p::ModelParameters)
        -> Tuple{Vector{Float64}, Float64, Int64}

Compute equilibrium market shares given firms’ positions on the quality ladder.

Each firm `i` is characterized by its ladder position `τᵢ = A_vec[i]` and
productivity

    Aᵢ = γ^{τᵢ},

where `γ` is the technology growth factor per innovation step. The set of
active firms is determined endogenously using `is_active`.

Let `ñ` denote the number of active firms and `Ã` the harmonic mean
productivity among them. The market share of firm `i` is given by

    sᵢ(A) =
        (1/σ) * [1 - ((ñ - σ)/ñ) * (Ã / Aᵢ)]   if i is active,
        0                                         otherwise.

This corresponds to the equilibrium expression

    sᵢ(𝐀ₜ) =
        (1/σ) [1 - ((ñ - σ)/ñ)(Ãₜ / Aᵢ,ₜ)]  for i ∈ 𝒩ᵖₜ,
        0                                       otherwise.

# Arguments
- `A_vec::Vector{Int64}`: Vector of firms’ positions on the quality ladder,
  where `A_vec[i] = τᵢ`.
- `p::ModelParameters`: Model parameters, including:
    - `γ`: innovation step size (technology growth factor per successful innovation),,
    - `σ`: risk aversion.

# Returns
- `Tuple{Vector{Float64}, Float64, Int64}`:
    - `share_vec::Vector{Float64}`: Equilibrium market shares for all firms
      (zero for inactive firms).
    - `Ã::Float64`: Harmonic mean productivity among active firms.
    - `ñ::Int64`: Number of active firms.

# Notes
- Market shares are strictly positive only for firms in the equilibrium
  active set.
- The active set satisfies the internal and external consistency conditions
  defined in the participation threshold characterization.
"""
function market_share(A_vec:: Vector{Int64}, p:: ModelParameters):: Tuple{Vector{Float64}, Float64, Int64}
    share_vec = zeros(length(A_vec))
    active_vec, Ã = is_active(A_vec, p)
    ñ = sum(active_vec)
    coef = (ñ - p.σ) / ñ * Ã
    for i in eachindex(A_vec)
        if active_vec[i]
            s_i = (1 - coef / (p.γ ^ (A_vec[i] - 1))) / p.σ
            share_vec[i] = s_i
        end
    end
    return share_vec, Ã, ñ
end;



"""
    competition_index(A_vec::Vector{Int64}, p::ModelParameters)
        -> Tuple{Float64, Vector{Float64}, Float64, Int64}

Compute the adjusted market competition–concentration index 𝒦(𝐀) and
associated equilibrium objects.

Given firms’ positions on the quality ladder, the function first computes
equilibrium market shares using `market_share`, obtaining:

- `share_vec`: equilibrium shares,
- `Ã`: harmonic mean productivity among active firms,
- `ñ`: number of active firms.

Let `HHI = ∑ᵢ sᵢ²` denote the Herfindahl–Hirschman index of concentration.
The adjusted competition–concentration index is computed using the
alternative representation

    𝒦(𝐀) =
        (σ / (ñ - σ)) * (1 - σ * HHI).

This expression highlights that 𝒦(𝐀) is a monotone transformation of the HHI
scaled by the endogenous number of active firms.

# Arguments
- `A_vec::Vector{Int64}`: Vector of firms’ ladder positions `τᵢ`.
- `p::ModelParameters`: Model parameters, including:
    - `γ`: innovation step size (technology growth factor per successful innovation),,
    - `σ`: risk aversion.

# Returns
- `Tuple{Float64, Vector{Float64}, Float64, Int64}`:
    - `K_value::Float64`: Adjusted competition–concentration index 𝒦(𝐀).
    - `share_vec::Vector{Float64}`: Equilibrium market shares.
    - `Ã::Float64`: Harmonic mean productivity among active firms.
    - `ñ::Int64`: Number of active firms.

# Notes
- 𝒦(𝐀) decreases with concentration (HHI) and incorporates endogenous
  participation through `ñ`.
- Shares of inactive firms are zero and therefore do not affect the HHI.
"""
function competition_index(A_vec:: Vector{Int64}, p:: ModelParameters):: Tuple{Float64, Vector{Float64}, Float64, Int64}
    share_vec, Ã, ñ = market_share(A_vec, p)
    HHI = sum(share_vec .^ 2) 
    result = (p.σ / (ñ - p.σ)) * (1 - p.σ * HHI)
    return result, share_vec, Ã, ñ
end;


"""
    adjusted_market_share(A_vec::Vector{Int64}, p::ModelParameters)
        -> Tuple{Vector{Float64}, Float64, Float64, Int64}

Compute adjusted market shares based on equilibrium shares and the
competition–concentration index.

The function first evaluates the competition index `K` and equilibrium
market shares `sᵢ` using `competition_index`. It then applies the
transformation

    s̃ᵢ = (σ² / (ñ * K)) * sᵢ²,

where:
- `σ` is the competition parameter,
- `ñ` is the number of active firms,
- `K` is the adjusted competition–concentration index.

Thus, adjusted shares are proportional to squared equilibrium shares,
rescaled by an endogenous factor depending on market structure.

# Arguments
- `A_vec::Vector{Int64}`: Vector of firms’ ladder positions `τᵢ`.
- `p::ModelParameters`: Model parameters, including:
    - `γ`: innovation step size (technology growth factor),
    - `σ`: risk aversion (competition) parameter.

# Returns
- `Tuple{Vector{Float64}, Float64, Int64}`:
    - `adjusted_shares::Vector{Float64}`: Adjusted market shares.
    - `K::Float64`: Adjusted competition–concentration index 𝒦(𝐀).
    - `Ã::Float64`: Harmonic mean productivity among active firms.
    - `ñ::Int64`: Number of active firms.

# Notes
- Inactive firms have zero equilibrium shares and therefore zero adjusted shares.
- The adjustment amplifies dispersion by squaring equilibrium shares and
  rescales them using the endogenous competition index.
"""
function adjusted_market_share(A_vec:: Vector{Int64}, p:: ModelParameters):: Tuple{Vector{Float64}, Float64, Float64, Int64}
    K, share_vec, Ã, ñ = competition_index(A_vec, p) 

    coef = (p.σ^2) / (ñ * K)
    convert_share(x) = coef * (x^2) 
    adjusted_shares = convert_share.(share_vec) 
    
    return adjusted_shares, K, Ã, ñ
end;



"""
    calculate_dividends(
        l_vec::Vector{Float64},
        s̃_vec::Vector{Float64},
        Ã::Float64,
        ñ::Int64,
        p::ModelParameters
    ) -> Vector{Float64}

Compute real dividends for each firm given labor choices and equilibrium
market outcomes.

Let `l_vec` denote firms’ R&D labor choices `lᵢʳ`, and let

    Lᴾ = 1 - ∑ᵢ lᵢʳ

be production labor. Given adjusted market shares `s̃ᵢ(𝐀)` and harmonic
mean productivity `Ã`, firm `i`’s real dividends are computed as

    Dᵢ =
        Ã * [ s̃ᵢ(𝐀) * (1 - ∑ⱼ lⱼʳ)
               - ((ñ - σ)/ñ) * lᵢʳ ],

where `ñ` is the number of active firms and `σ` is the competition parameter.

# Arguments
- `l_vec::Vector{Float64}`: Vector of firms’ R&D labor choices `lᵢʳ`.
- `s̃_vec::Vector{Float64}`: Adjusted market shares `s̃ᵢ(𝐀)`.
- `Ã::Float64`: Harmonic mean productivity among active firms.
- `ñ::Int64`: Number of active firms.
- `p::ModelParameters`: Model parameters, including:
    - `σ`: competition parameter.

# Returns
- `Vector{Float64}`: Vector of real dividends for each firm.
"""
function calculate_dividends(l_vec::Vector{Float64}, s̃_vec:: Vector{Float64}, Ã:: Float64, ñ:: Int64, p:: ModelParameters) 
    Lᴾ = 1 - sum(l_vec)
    d_vec = ((s̃_vec .* Lᴾ) .- ((ñ - p.σ) / ñ) .* l_vec) .* Ã
    return d_vec
end;


"""
    calculate_consumption(
        l_vec::Vector{Float64},
        K::Float64,
        Ã::Float64,
        ñ::Int64,
        p::ModelParameters
    ) -> Float64

    calculate_consumption(
        Lᴿ::Float64,
        K::Float64,
        Ã::Float64,
        ñ::Int64,
        p::ModelParameters
    ) -> Float64

Compute aggregate real consumption given equilibrium market conditions
and R&D labor allocation.

Let total R&D labor be either:

    Lᴿ = ∑ᵢ lᵢʳ          (vector version),  or
    Lᴿ                    (scalar version).

Production labor is

    Lᴾ = 1 - Lᴿ.

Aggregate real consumption is given by

    C = (σ / ñ) * (Lᴾ / K) * Ã,

where:
- `σ` is the competition parameter,
- `ñ` is the number of active firms,
- `K` is the adjusted competition–concentration index,
- `Ã` is the harmonic mean productivity among active firms.

The two method definitions differ only in whether R&D labor is provided
at the firm level (`l_vec`) or in aggregate (`Lᴿ`).

# Arguments
- `l_vec::Vector{Float64}`: Firm-level R&D labor allocations.
- `Lᴿ::Float64`: Aggregate R&D labor.
- `K::Float64`: Adjusted competition–concentration index.
- `Ã::Float64`: Harmonic mean productivity among active firms.
- `ñ::Int64`: Number of active firms.
- `p::ModelParameters`: Model parameters, including:
    - `σ`: competition parameter.

# Returns
- `Float64`: Aggregate real consumption.

# Notes
- Consumption decreases with R&D labor through the resource constraint
  `Lᴾ = 1 - Lᴿ`.
- The expression scales production by both market structure (`K`, `ñ`)
  and average productivity (`Ã`).
"""
function calculate_consumption(l_vec::Vector{Float64}, K:: Float64, Ã:: Float64, ñ:: Int64, p:: ModelParameters):: Float64
    Lᴾ = 1.0 - sum(l_vec)
    return (p.σ / ñ) * (Lᴾ / K) * Ã
end;

function calculate_consumption(Lᴿ::Float64, K:: Float64, Ã:: Float64, ñ:: Int64, p:: ModelParameters):: Float64
    Lᴾ = 1.0 - Lᴿ
    return (p.σ / ñ) * (Lᴾ / K) * Ã
end;



"""
    innovation_success_prob(l::Float64, p::ModelParameters) -> Float64

Compute the probability of successful innovation as a function of R&D labor.

Given R&D labor `l = lᵢʳ`, the firm’s technology evolves according to

    Aᵢ,ₜ₊₁ =
        γ Aᵢ,ₜ    with probability  η(lᵢʳ),
        Aᵢ,ₜ       with probability  1 - η(lᵢʳ),

where `γ` is the innovation step size and `η(·)` is the innovation
success probability, satisfying ∂η/∂l ≥ 0 whenever the specification
depends on `l`.

The functional form of `η(l)` is determined by `p.PROB`:

- `"SQRT"`:     η(l) = √(α l)
- `"EXP"`:      η(l) = 1 - exp(-α l)
- `"LINEAR"`:   η(l) = α l
- `"CONS"`:     η(l) = α
- `"ZERO"`:     η(l) = 0

In all cases except `"ZERO"`, the probability is clamped to the interval `[0, 1]`.

# Arguments
- `l::Float64`: R&D labor hired by the firm.
- `p::ModelParameters`: Model parameters, including:
    - `α`: parameter governing innovation effectiveness,
    - `γ`: innovation step size,
    - `PROB`: functional form specification for η(·).

# Returns
- `Float64`: Innovation success probability η(l).

# Notes
- The function guarantees that probabilities lie in `[0, 1]`.
- Under `"CONS"` and `"ZERO"`, innovation is independent of R&D labor.
"""
function innovation_success_prob(l:: Float64, p:: ModelParameters)
    if p.PROB == "SQRT"
        return clamp(sqrt(l * p.α), 0.0, 1.0)
    elseif p.PROB == "EXP"
        return clamp(1 - exp(-l * p.α), 0.0, 1.0)
    elseif p.PROB == "LINEAR"
        return clamp(l * p.α, 0.0, 1.0)
    elseif p.PROB == "CONS" 
        return clamp(p.α, 0.0, 1.0)
    elseif p.PROB == "ZERO" 
        return 0.0
    else
        error("Unknown PROB: $(p.PROB)")
    end
end;