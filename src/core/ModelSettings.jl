"""
    ModelSettings(; kwargs...)

Primary container for the SchumpeterCournot model.

Stores:
- Structural parameters
- State space indexing
- Value and policy grids
- Solver tolerances and damping controls

The object is mutated during solution.
"""
mutable struct ModelSettings 
    # model parameters 
    param:: ModelParameters 
    
    # state space 
    τ_max:: Int64 
    
    # policy and value grids 
    V_grid:: Array{Float64, 2} 
    policy_grid:: Array{Float64, 2} 
    policy_grid_j:: Array{Float64, 2} 
    
    # operational parameters 
    max_iter_PF_update:: Int64 
    tol_PF_update:: Float64 
    clamp_rate_PF_solver:: Float64 
    clamp_rate_PF_update:: Float64 
    sdf_relaxer:: Float64 
    
    # indexing variables 
    idx_map:: TupleIndexMap 
    
    # parameter control 
    l_max:: Float64 
end;


"""
    ModelSettings(; ...)

Construct and initialize model instance.
Allocates grids and builds index mapping.
"""
function ModelSettings(;
    # model parameters
    param:: ModelParameters = ModelParameters(),
    
    # state space
    τ_max:: Int64 = 50,

    # initial policy_j
    pfunc_j:: Float64 = 1e-3,

    # operational parameters
    max_iter_PF_update:: Int64 = 500,
    tol_PF_update:: Float64 = 1e-5,
    clamp_rate_PF_solver:: Float64 = 0.5,
    clamp_rate_PF_update:: Float64 = 0.5,
    sdf_relaxer:: Float64 = 1.0,

    # parameter control
    l_max:: Float64 = 0.3
)
    N = param.n
    index_map = build_lookup(N-1, τ_max)

    dimentions = (τ_max, length(index_map.idx_to_tuple))
    

    # initialize policy and value grids
    V_grid = zeros(Float64, dimentions...)
    policy_grid = zeros(Float64, dimentions...)
    policy_grid_j = ones(Float64, dimentions...) * pfunc_j


    return ModelSettings(
        param, τ_max, V_grid, policy_grid, policy_grid_j, 
        max_iter_PF_update, 
        tol_PF_update, 
        clamp_rate_PF_solver, clamp_rate_PF_update, sdf_relaxer, 
        index_map,
        l_max
    )
end;