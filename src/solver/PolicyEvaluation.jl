# ============================================================
# Policy Evaluation
# ============================================================

"""
    solve_values!(model)

Solve linear Bellman system under fixed policy.
Updates `model.V_grid` in-place.
"""
function solve_values!(model:: DSCIModel)
    A, b = construct_vf_equation_system(model)
    s = A \ b
    V_grid = reshape(s, (:, model.env.τ_max))'
    model.state.V_grid .= V_grid
    return
end;