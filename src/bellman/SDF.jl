"""
    calculate_sdf(model, ...)

Compute stochastic discount factor between two states.
Includes optional relaxation for numerical stability.
"""
function calculate_sdf(
    model:: ModelSettings,
    Lᴿₜ::Float64, Lᴿₜ₊₁::Float64,
    Kₜ:: Float64, Kₜ₊₁:: Float64,
    Ãₜ:: Float64, Ãₜ₊₁:: Float64,
    ñₜ:: Int, ñₜ₊₁:: Int,
):: Float64
    cₜ   = calculate_consumption(Lᴿₜ,    Kₜ,     Ãₜ,     ñₜ,     model.param)
    cₜ₊₁ = calculate_consumption(Lᴿₜ₊₁,  Kₜ₊₁,   Ãₜ₊₁,   ñₜ₊₁,   model.param) 

    sdf = (cₜ₊₁ / cₜ) ^ (-1 * model.param.σ)
    sdf = (1.0 - model.sdf_relaxer) + model.sdf_relaxer * sdf
    return sdf * model.param.β
end;
