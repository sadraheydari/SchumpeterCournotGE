struct ModelParameters
    n::Int64        # number of firms
    β::Float64      # discount factor
    σ::Float64      # risk aversion
    γ::Float64      # innovation step size
    α::Float64      # probability of successful innovation parameter
    PROB::String    # innovation probability function
end;


function ModelParameters(;
    n::Int64 = 2,
    β::Float64 = 0.9,
    σ::Float64 = 1.5,
    γ::Float64 = 1.04,
    α::Float64 = 5.0,
    PROB::String = "SQRT"    
)
    return ModelParameters(n, β, σ, γ, α, PROB)
end;