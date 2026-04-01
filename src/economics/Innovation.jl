
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
function innovation_success_prob_l(l:: Float64, p:: ModelParameters)
    if p.PROB == "SQRT"
        return clamp(sqrt(l * p.α), 0.0, 1.0)
    elseif p.PROB == "SQRT-ADJ"
        return clamp(sqrt(l * p.α + 1.0) - 1.0, 0.0, 1.0)
    elseif p.PROB == "EXP"
        return clamp(1 - exp(-l * p.α), 0.0, 1.0)
    elseif p.PROB == "LINEAR"
        return clamp(l * p.α, 0.0, 1.0)
    elseif p.PROB == "CONS" 
        return clamp(p.α, 0.0, 1.0)
    elseif p.PROB == "ZERO" 
        return 0.0
    elseif p.PROB == "PWR1"
        return clamp((l*p.α)^(0.1), 0.0, 1.0)
    elseif p.PROB == "PWR2"
        return clamp((l*p.α)^(0.2), 0.0, 1.0)
    elseif p.PROB == "PWR3"
        return clamp((l*p.α)^(0.3), 0.0, 1.0)
    elseif p.PROB == "PWR4"
        return clamp((l*p.α)^(0.4), 0.0, 1.0)
    elseif p.PROB == "PWR6"
        return clamp((l*p.α)^(0.6), 0.0, 1.0)
    elseif p.PROB == "PWR8"
        return clamp((l*p.α)^(0.8), 0.0, 1.0)
    else
        error("Unknown PROB: $(p.PROB)")
    end
end;
innovation_success_prob_l(l:: Float64, e:: ModelEnvironment) = innovation_success_prob_l(l, e.param);
innovation_success_prob_l(l:: Float64, m:: DSCIModel) = innovation_success_prob_l(l, m.env);



function innovation_success_prob_A(l:: Float64, A:: Int64, p:: ModelParameters)
    if p.PROB == "LINEAR"
        return clamp(l * p.α * (p.γ ^ A), 0.0, 1.0)
    elseif p.PROB == "EXP"
        return clamp(1 - exp(-l * p.α / (p.γ ^ A)), 0.0, 1.0)
    elseif p.PROB == "LOGIT"
        return clamp(l / ((p.γ ^ A) + l), 0.0, 1.0)
    else
        error("PROB=$(p.PROB) does not support A-dependence")
    end
end;
innovation_success_prob_A(l:: Float64, A:: Int64, e:: ModelEnvironment) = innovation_success_prob_A(l, A, e.param);
innovation_success_prob_A(l:: Float64, A:: Int64, m:: DSCIModel) = innovation_success_prob_A(l, A, m.env);

# Convenience wrapper to dispatch based on innovation type

innovation_success_prob(::NonChanging, l:: Float64, A:: Int64, p:: ModelParameters) = innovation_success_prob_l(l, p)
innovation_success_prob(::Decreasing, l:: Float64, A:: Int64, p:: ModelParameters) = innovation_success_prob_A(l, A, p)

innovation_success_prob(l:: Float64, A:: Int64, e:: ModelEnvironment) = innovation_success_prob(e.INNOV_TYPE, l, A, e.param)
innovation_success_prob(l:: Float64, A:: Int64, m:: DSCIModel) = innovation_success_prob(m.env.INNOV_TYPE, l, A, m.env.param)