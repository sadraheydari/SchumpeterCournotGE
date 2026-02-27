# ============================================================
# Policy Improvement
# ============================================================

"""
    update_policy!(model)

Given value function, compute optimal R&D investment
via one-dimensional optimization at each state.
"""
function update_policy!(model:: DSCIModel; clamp:: Bool = false, clamp_rate:: Float64 = 0.2)
    τ_max = model.env.τ_max

    @threads for A_1 in 1:τ_max
        for (local_idx, state) in enumerate(model.env.idx_map.idx_to_tuple)
            
            A_vec = [A_1; state]
            _, A_old_idx, v_old_idx = flat_idx(model.env, A_1, local_idx)

            s̃ₜ, Kₜ, Ãₜ, ñₜ = adjusted_market_share(A_vec, model.env.param) 
            current_market_vars = (Kₜ, Ãₜ, ñₜ)
            base_l_vec = get_labour_demand(A_vec, model)

            # objective function for optimization
            function objective(x:: Float64)
                l_vec = copy(base_l_vec)
                l_vec[1] = x

                L_old = sum(l_vec)
                d_vec = calculate_dividends(l_vec, s̃ₜ, Ãₜ, ñₜ, model.env.param)

                # Current dividends
                expected_val = scale_flow(
                    model.settings.value_scaling, # Scaling mode
                    d_vec[1],                     # Firm 1's dividend
                    model.env.param.γ ^ (A_1 - 1) # Scale by productivity level if using detrended values
                )
                                
                η_func = x -> innovation_success_prob(x, model.env.param)
                η_vec = η_func.(l_vec)

                for transition_state in get_transition_states(model.env.param)
                    pr = calculate_transition_probability(transition_state, η_vec)
                    A_new = A_vec .+ transition_state

                    term_value = 0.0
                    for (_, A_idx, v_idx, weighted_sdf) in get_transition_contributions(model, A_new, A_vec, L_old, current_market_vars)
                        term_value += weighted_sdf * model.state.V_grid[A_idx, v_idx]
                    end
                    expected_val += term_value * pr
                end

                return -expected_val
            end;

            # optimize the objective function
            result = optimize(objective, 0.0, model.env.l_max)
            optimal_l = result.minimizer
            if clamp
                optimal_l = clamp_rate * optimal_l + (1 - clamp_rate) * model.state.policy_grid[A_old_idx, v_old_idx]
            end
            model.state.policy_grid[A_old_idx, v_old_idx] = optimal_l
        end
    end
end;
