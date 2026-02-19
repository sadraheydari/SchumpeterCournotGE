


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
end


mutable struct SolverState
    V_grid::Matrix{Float64}
    policy_grid::Matrix{Float64}
    policy_grid_j::Matrix{Float64}
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
    l_max = 0.3
)
    idx_map = build_lookup(param.n - 1, τ_max)
    
    env = ModelEnvironment(param, τ_max, idx_map, l_max)
    
    settings = SolverSettings(max_iter_update, tol_update, clamp_rate_update, sdf_relaxer)
    
    dimentions = (τ_max, length(idx_map.idx_to_tuple))
    state = SolverState(zeros(dimentions...), zeros(dimentions...), ones(dimentions...) * initial_pj)
    
    return DSCIModel(env, settings, state)
end;




macro param_forward(def)
    @assert def.head == :function "Use @param_forward with a function definition"

    sig  = def.args[1]
    body = def.args[2]

    fname = sig.args[1]
    args  = sig.args[2:end]

    # Find the argument typed as ModelParameters
    idx = findfirst(arg -> (
        arg isa Expr &&
        arg.head == :(::) &&
        arg.args[2] == :ModelParameters
    ), args)

    @assert idx !== nothing "Function must have an argument typed as ModelParameters"

    # Extract argument name
    param_arg = args[idx].args[1]

    # Create modified signatures
    args_env = copy(args)
    args_env[idx] = :($param_arg::ModelEnvironment)

    args_model = copy(args)
    args_model[idx] = :($param_arg::DSCIModel)

    # Build argument list for forwarding call
    call_args_env = [
        i == idx ? :($param_arg.param) : args[i].args[1]
        for i in eachindex(args)
    ]

    call_args_model = [
        i == idx ? :($param_arg.env.param) : args[i].args[1]
        for i in eachindex(args)
    ]

    quote
        # Original method
        $def

        # Forward ModelEnvironment
        function $fname($(args_env...))
            $fname($(call_args_env...))
        end

        # Forward DSCIModel
        function $fname($(args_model...))
            $fname($(call_args_model...))
        end
    end |> esc
end




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