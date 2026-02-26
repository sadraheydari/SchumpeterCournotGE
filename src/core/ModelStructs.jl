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


struct ModelEnvironment
    param::ModelParameters
    τ_max::Int
    idx_map::TupleIndexMap
    l_max::Float64
end


struct SolverSettings
    max_iter_update::Int
    tol_update::Float64
    clamp_rate_update::Float64
    sdf_relaxer::Float64
    value_scaling::ValueScaling
end


function SolverSettings(
    max_iter_update::Int,
    tol_update::Float64,
    clamp_rate_update::Float64,
    sdf_relaxer::Float64
)
    return SolverSettings(max_iter_update, tol_update, clamp_rate_update, sdf_relaxer, Levels())
end


mutable struct SolverState
    V_grid::Matrix{Float64}
    policy_grid::Matrix{Float64}
    policy_grid_j::Matrix{Float64}
    is_converged::Bool
end


"""
    DSCIModel
DSCI Model (Dynamic Strategic Competition & Innovation)
"""
mutable struct DSCIModel
    env::ModelEnvironment
    settings::SolverSettings
    state::SolverState
end


function DSCIModel(;
    param = ModelParameters(),
    τ_max = 50,
    initial_pj = 1e-3,
    max_iter_update = 500,
    tol_update = 1e-5,
    clamp_rate_update = 0.5,
    sdf_relaxer = 1.0,
    l_max = 0.3,
    value_scaling = Levels()
)
    idx_map = build_lookup(param.n - 1, τ_max)
    
    env = ModelEnvironment(param, τ_max, idx_map, l_max)
    
    settings = SolverSettings(max_iter_update, tol_update, clamp_rate_update, sdf_relaxer, value_scaling)
    
    dimentions = (τ_max, length(idx_map.idx_to_tuple))
    state = SolverState(
        zeros(dimentions...), 
        zeros(dimentions...), 
        ones(dimentions...) * initial_pj,
        false
    )
    
    return DSCIModel(env, settings, state)
end;