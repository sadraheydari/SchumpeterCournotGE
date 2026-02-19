"""
    calculate_sdf(model, ...)

Compute stochastic discount factor between two states.
Includes optional relaxation for numerical stability.
"""
function calculate_sdf(
    model:: DSCIModel,
    Lᴿₜ::Float64, Lᴿₜ₊₁::Float64,
    Kₜ:: Float64, Kₜ₊₁:: Float64,
    Ãₜ:: Float64, Ãₜ₊₁:: Float64,
    ñₜ:: Int, ñₜ₊₁:: Int,
):: Float64
    cₜ   = calculate_consumption(Lᴿₜ,    Kₜ,     Ãₜ,     ñₜ,     model.env.param)
    cₜ₊₁ = calculate_consumption(Lᴿₜ₊₁,  Kₜ₊₁,   Ãₜ₊₁,   ñₜ₊₁,   model.env.param) 

    sdf = (cₜ₊₁ / cₜ) ^ (-1 * model.env.param.σ)
    sdf = (1.0 - model.settings.sdf_relaxer) + model.settings.sdf_relaxer * sdf
    return sdf * model.env.param.β
end;
