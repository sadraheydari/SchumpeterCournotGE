"""
    construct_vf_equation_system(model)

Construct sparse linear system:

    A * V = b

corresponding to the Bellman equation under fixed policies.
"""
function construct_vf_equation_system(model:: DSCIModel)
    τ_max = model.env.τ_max
    n = model.env.param.n
    len_state_space = length(model.env.idx_map.idx_to_tuple)
    total_states = τ_max * len_state_space
    
    i_list = Int[];     sizehint!(i_list, total_states * (2^n + 4n))
    j_list = Int[];     sizehint!(j_list, total_states * (2^n + 4n))
    v_list = Float64[]; sizehint!(v_list, total_states * (2^n + 4n))

    b_vec = zeros(Float64, total_states)

    for A_1 in 1:τ_max
        for (local_idx, state) in enumerate(model.env.idx_map.idx_to_tuple)
            A_old = [A_1; state]
            idx_old, _, _ = flat_idx(model.env, A_1, local_idx)

            i_list = push!(i_list, idx_old) 
            j_list = push!(j_list, idx_old)
            v_list = push!(v_list, 1.0)

            s̃ₜ, Kₜ, Ãₜ, ñₜ = adjusted_market_share(A_old, model.env.param) 
            l_vec = get_labour_demand(A_old, model)            
            L_old = sum(l_vec)

            # RHS: dividends
            d_vec = calculate_dividends(l_vec, s̃ₜ, Ãₜ, ñₜ, model.env.param)
            b_vec[idx_old] = scale_flow(
                model.settings.value_scaling, # Scaling mode
                d_vec[1],                     # Firm 1's dividend
                model.env.param.γ ^ (A_1 - 1) # Scale by productivity level if using detrended values
            )
            
            # LHS: Transitions
            η_vec = [innovation_success_prob(l, model.env.param) for l in l_vec]

            for transition_state in get_transition_states(model)
                pr = calculate_transition_probability(transition_state, η_vec)
                A_new = A_old .+ transition_state
                for (idx_target, _, _, weighted_sdf) in get_transition_contributions(model, A_new, A_old, L_old, (Kₜ, Ãₜ, ñₜ))
                    i_list = push!(i_list, idx_old)
                    j_list = push!(j_list, idx_target)
                    v_list = push!(v_list, -pr * weighted_sdf)
                end
            end
        end
    end
    return sparse(i_list, j_list, v_list), b_vec
end;