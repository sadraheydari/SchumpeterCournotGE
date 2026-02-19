using SparseArrays

# ============================================================
# Core Economic Objects
# ============================================================

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


"""
    flat_idx(model, A₁, state)

Map structured state into flattened index for Bellman system.
"""
function flat_idx(model:: ModelSettings, A_1:: Int, state:: Vector{Int})
    local_idx = @sget model.idx_map.tuple_to_idx[state]
    global_idx = local_idx + (A_1 - 1) * length(model.idx_map.idx_to_tuple)
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

function get_transition_states(m:: ModelSettings)
    return get_transition_states(m.param)
end;


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
function get_labour_demand(A_vec:: Vector{Int64}, model:: ModelSettings ):: Vector{Float64} 
    
    l_vec = zeros(Float64, length(A_vec))
    
    # the A_1 is the demand for labour from `policy_grid`
    @views state = A_vec[2:end]
    idx = @sget model.idx_map.tuple_to_idx[state]
    A_1 = A_vec[1]
    l_vec[1] = model.policy_grid[A_1, idx]

    # use `policy_grid_j` for the others
    A_vec = copy(A_vec)
    A_tm1 = A_vec[1]
    for j in 2:length(A_vec)
        A_vec[j-1], A_vec[j], A_tm1 = A_tm1, A_vec[1], A_vec[j]
        @views state = A_vec[2:end]
        idx = @sget model.idx_map.tuple_to_idx[state]
        l_vec[j] = model.policy_grid_j[A_tm1, idx]
    end

    return l_vec
end;


"""
    get_transition_contributions(model, A_new, L_old, market_vars)

Return continuation value contributions for given innovation realization.
"""
function get_transition_contributions(
    model:: ModelSettings, 
    A_new:: Vector{Int}, 
    L_old:: Float64, 
    current_market_vars:: Tuple{Float64, Float64, Int}
    )
    τ_max = model.τ_max
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
        idx_target, A_1, local_idx = flat_idx(model, pt_state[1], pt_state[2:end])

        K_next, _, Ã_next, ñ_next = competition_index(pt_state, model.param)
        l_vec_next = get_labour_demand(pt_state, model)
        L_next = sum(l_vec_next)

        sdf = calculate_sdf(model, L_old, L_next, Kₜ, K_next, Ãₜ, Ã_next, ñₜ, ñ_next)
        (idx_target, A_1, local_idx, sdf * weight)
    end for (pt_state, weight) in points)
end;


# ============================================================
# Bellman Linear System
# ============================================================

"""
    construct_vf_equation_system(model)

Construct sparse linear system:

    A * V = b

corresponding to the Bellman equation under fixed policies.
"""
function construct_vf_equation_system(model:: ModelSettings)
    τ_max = model.τ_max
    n = model.param.n
    len_state_space = length(model.idx_map.idx_to_tuple)
    total_states = τ_max * len_state_space
    
    i_list = Int[];     sizehint!(i_list, total_states * (2^n + 4n))
    j_list = Int[];     sizehint!(j_list, total_states * (2^n + 4n))
    v_list = Float64[]; sizehint!(v_list, total_states * (2^n + 4n))

    b_vec = zeros(Float64, total_states)

    for A_1 in 1:τ_max
        for (local_idx, state) in enumerate(model.idx_map.idx_to_tuple)
            A_old = [A_1; state]
            idx_old, _, _ = flat_idx(model, A_1, state)

            i_list = push!(i_list, idx_old) 
            j_list = push!(j_list, idx_old)
            v_list = push!(v_list, 1.0)

            s̃ₜ, Kₜ, Ãₜ, ñₜ = adjusted_market_share(A_old, model.param) 
            l_vec = get_labour_demand(A_old, model)            
            L_old = sum(l_vec)

            # RHS: dividends
            d_vec = calculate_dividends(l_vec, s̃ₜ, Ãₜ, ñₜ, model.param)
            b_vec[idx_old] = d_vec[1]
            
            # LHS: Transitions
            η_vec = [innovation_success_prob(l, model.param) for l in l_vec]

            for transition_state in get_transition_states(model)
                pr = calculate_transition_probability(transition_state, η_vec)
                A_new = A_old .+ transition_state
                for (idx_target, _, _, weighted_sdf) in get_transition_contributions(model, A_new, L_old, (Kₜ, Ãₜ, ñₜ))
                    i_list = push!(i_list, idx_old)
                    j_list = push!(j_list, idx_target)
                    v_list = push!(v_list, -pr * weighted_sdf)
                end
            end
        end
    end
    return sparse(i_list, j_list, v_list), b_vec
end;