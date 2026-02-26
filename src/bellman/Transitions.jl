"""
    flat_idx(env, A₁, state)

Map structured state into flattened index for Bellman system.
"""
function flat_idx(env:: ModelEnvironment, A_1:: Int, state_idx:: Int)
    global_idx = state_idx + (A_1 - 1) * length(env.idx_map.idx_to_tuple)
    return global_idx, A_1, state_idx
end;

function flat_idx(env:: ModelEnvironment, A_1:: Int, state:: Vector{Int})
    local_idx = @sget env.idx_map.tuple_to_idx[state]
    global_idx = local_idx + (A_1 - 1) * length(env.idx_map.idx_to_tuple)
    return global_idx, A_1, local_idx
end;




"""
    get_transition_states(model)

Iterator over all innovation outcome vectors.
"""
function get_transition_states(p:: ModelParameters)
    base_iter = Iterators.product(ntuple(_ -> [0, 1], p.n)...)
    return (collect(t) for t in base_iter)
end;
get_transition_states(e:: ModelEnvironment) = get_transition_states(e.param);
get_transition_states(m:: DSCIModel) = get_transition_states(m.env.param);


"""
    calculate_transition_probability(state, η)

Return probability of innovation outcome.
"""
function calculate_transition_probability(
    transition_state:: Vector{Int64}, success_prob:: Vector{Float64}
):: Float64
    @assert length(transition_state) == length(success_prob) "Length of transition state and success probability vectors must be the same."
    
    prob = 1.0
    for i in eachindex(transition_state)
        @inbounds sᵢ = transition_state[i]
        @inbounds ηᵢ = success_prob[i]
        prob *=  ηᵢ * sᵢ + (1 - ηᵢ) * (1 - sᵢ) 
    end
    return prob
end;


"""
    get_labour_demand(A_vec, model)

Return labor allocation vector given policy grids.
"""
function get_labour_demand(A_vec:: Vector{Int64}, model:: DSCIModel):: Vector{Float64} 
    
    l_vec = zeros(Float64, length(A_vec))
    
    # the A_1 is the demand for labour from `policy_grid`
    @views state = A_vec[2:end]
    idx = @sget model.env.idx_map.tuple_to_idx[state]
    A_1 = A_vec[1]
    l_vec[1] = model.state.policy_grid[A_1, idx]

    # use `policy_grid_j` for the others
    A_vec = copy(A_vec)
    A_tm1 = A_vec[1]
    for j in 2:length(A_vec)
        A_vec[j-1], A_vec[j], A_tm1 = A_tm1, A_vec[1], A_vec[j]
        @views state = A_vec[2:end]
        idx = @sget model.env.idx_map.tuple_to_idx[state]
        l_vec[j] = model.state.policy_grid_j[A_tm1, idx]
    end

    return l_vec
end;


"""
    get_transition_contributions(model, A_new, L_old, market_vars)

Return continuation value contributions for given innovation realization.
"""
function get_transition_contributions(
    model:: DSCIModel,
    A_new:: Vector{Int},
    A_old:: Vector{Int},
    L_old:: Float64, 
    current_market_vars:: Tuple{Float64, Float64, Int}
    )
    τ_max = model.env.τ_max
    (Kₜ, Ãₜ, ñₜ) = current_market_vars

    # Extrapolte the n+1 state using a linear approximation 
    # F_{n+1} = F_n + (F_n - F_{n-1}) = 2 * F_n - F_{n-1}
    if any(A_new .> τ_max)
        A_n     = clamp.(A_new, 1, τ_max)
        diff    = A_new .- A_n
        A_nm1   = A_n .- diff
        points = ((A_n, 2.0), (A_nm1, -1.0))
    else
        points = ((A_new, 1.0),)
    end

    return (begin
        idx_target, A_1, local_idx = flat_idx(model.env, pt_state[1], pt_state[2:end])

        K_next, _, Ã_next, ñ_next = competition_index(pt_state, model.env.param)
        l_vec_next = get_labour_demand(pt_state, model)
        L_next = sum(l_vec_next)

        sdf = calculate_sdf(model, L_old, L_next, Kₜ, K_next, Ãₜ, Ã_next, ñₜ, ñ_next)
        scaled_sdf = scale_kernel(
            model.settings.value_scaling, # Scaling mode
            sdf,                          # Unscaled SDF
            model.env.param.γ ^ (A_old[1] - 1),
            model.env.param.γ ^ (pt_state[1] - 1)
        )
        (idx_target, A_1, local_idx, scaled_sdf * weight)
    end for (pt_state, weight) in points)
end;