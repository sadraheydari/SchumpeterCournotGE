# ============================================================
# Policy Evaluation
# ============================================================

"""
    solve_values!(model)

Solve linear Bellman system under fixed policy.
Updates `model.V_grid` in-place.
"""
function solve_values!(model:: ModelSettings)
    A, b = construct_vf_equation_system(model)
    s = A \ b
    V_grid = reshape(s, (:, model.τ_max))'
    model.V_grid .= V_grid
    return
end;


# ============================================================
# Policy Improvement
# ============================================================

"""
    update_policy!(model)

Given value function, compute optimal R&D investment
via one-dimensional optimization at each state.
"""
function update_policy!(model:: ModelSettings)
    τ_max = model.τ_max

    @threads for A_1 in 1:τ_max
        for (local_idx, state) in enumerate(model.idx_map.idx_to_tuple)
            
            A_vec = [A_1; state]
            idx_old, A_old_idx, v_old_idx = flat_idx(model, A_1, state)

            s̃ₜ, Kₜ, Ãₜ, ñₜ = adjusted_market_share(A_vec, model.param) 
            current_market_vars = (Kₜ, Ãₜ, ñₜ)
            base_l_vec = get_labour_demand(A_vec, model)

            # objective function for optimization
            function objective(x:: Float64)
                l_vec = copy(base_l_vec)
                l_vec[1] = x

                L_old = sum(l_vec)
                d_vec = calculate_dividends(l_vec, s̃ₜ, Ãₜ, ñₜ, model.param)

                # Current dividends
                expected_val = d_vec[1]

                η_vec = [innovation_success_prob(l, model.param) for l in l_vec]

                for transition_state in get_transition_states(model)
                    pr = calculate_transition_probability(transition_state, η_vec)
                    A_new = A_vec .+ transition_state

                    term_value = 0.0
                    for (idx_target, A_idx, v_idx, weighted_sdf) in get_transition_contributions(model, A_new, L_old, current_market_vars)
                        term_value += weighted_sdf * model.V_grid[A_idx, v_idx]
                    end
                    expected_val += term_value * pr
                end

                return -expected_val
            end

            # optimize the objective function
            result = optimize(objective, 0.0, model.l_max)
            model.policy_grid[A_old_idx, v_old_idx] = result.minimizer
        end
    end;
end;


# ============================================================
# Policy Function Iteration
# ============================================================

"""
    solve_PFI!(model; showprgs=true)

Solve the SchumpeterCournot model using Policy Function Iteration.

Algorithm:
1. Policy evaluation
2. Policy improvement
3. Damped update
4. Convergence check
"""
function solve_PFI!(model::ModelSettings; showprgs=true)

    tracker = showprgs ?
        ProgressTracker(model.tol_PF_update) :
        nothing

    for iter in 1:model.max_iter_PF_solver

        temp = copy(model.policy_grid_j)

        solve_values!(model)
        update_policy!(model)

        diff = maximum(abs.(model.policy_grid .- temp))

        showprgs && update!(tracker, diff, iter)

        if diff < model.tol_PF_update
            showprgs && finish!(tracker, iter, diff)
            break
        end

        model.policy_grid_j .=
            (1 - model.clamp_rate_PF_solver) .* model.policy_grid_j .+
             model.clamp_rate_PF_solver      .* model.policy_grid
    end
end;