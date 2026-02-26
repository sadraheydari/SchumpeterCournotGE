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
function solve_PFI!(model:: DSCIModel; showprgs=true)

    tracker = showprgs ?
        ProgressBar(model.settings.max_iter_update,model.settings.tol_update) :
        nothing

    for iter in 1:model.settings.max_iter_update

        temp = copy(model.state.policy_grid_j)

        solve_values!(model)
        update_policy!(model)

        diff = maximum(abs.(model.state.policy_grid .- temp))

        showprgs && update!(tracker, diff, iter)

        if diff < model.settings.tol_update
            showprgs && finish!(tracker, iter, diff)
            model.state.is_converged = true
            break
        end

        model.state.policy_grid_j .=
            (1 - model.settings.clamp_rate_update) .* model.state.policy_grid_j .+
             model.settings.clamp_rate_update      .* model.state.policy_grid
    end
end;