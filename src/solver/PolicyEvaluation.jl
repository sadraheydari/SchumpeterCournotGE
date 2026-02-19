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