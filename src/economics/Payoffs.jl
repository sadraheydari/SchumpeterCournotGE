
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
