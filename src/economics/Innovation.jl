
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
@param_forward function innovation_success_prob(l:: Float64, p:: ModelParameters)
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