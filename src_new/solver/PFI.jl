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
        ProgressBar(model.tol_PF_update) :
        nothing

    for iter in 1:model.max_iter_PF_update

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