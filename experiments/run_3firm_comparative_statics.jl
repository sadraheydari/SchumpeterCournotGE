using Pkg
Pkg.activate(".")

using SchumpeterCournot
using LinearAlgebra, Statistics

# --- 1. Baseline Configuration ---
const N_FIRMS = 3
const TAU_MAX = 60
const L_MAX   = 0.3

baseline_params = (
    σ = 1.3, 
    γ = 1.03, 
    β = 0.93, 
    α = 5.0
)

# --- 2. Define Parameter Sweeps ---
# We vary one at a time, keeping others at baseline
sweeps = [
    (:σ, [1.2, 1.4]),
    (:γ, [1.01, 1.05]),
    (:β, [0.91, 0.95]),
    (:α, [1.0, 10.0])
]

# --- 3. Initialize Model Once ---
# This allocates the large grids in SolverState
model = DSCIModel(
    param = ModelParameters(
        n = N_FIRMS,
        σ = baseline_params.σ,
        γ = baseline_params.γ,
        β = baseline_params.β,
        α = baseline_params.α
    ),
    τ_max = TAU_MAX,
    l_max = L_MAX,
    clamp_rate_update = 0.1,
    sdf_relaxer = 1.0
)

# Helper function to update structural environment in-place
function update_model_params!(m::DSCIModel, new_p::ModelParameters)
    # Reconstruct the Environment struct using existing fixed components
    # This replaces the 'env' pointer but keeps 'state' (the grids) intact
    m.env = ModelEnvironment(new_p, m.env.τ_max, m.env.idx_map, m.env.l_max)
    m.state.is_converged = false # reset convergence status for new parameters
end

# --- 4. Execution Loop ---

println("Starting Baseline Run...")
SchumpeterCournot.solve_PFI!(model)
save_model(model, dir="saved_models/3-firm") # save_model handles directory/naming internally

println("Starting Comparative Statics Sweep...")

for (param_name, values) in sweeps
    for val in values
        println("Solving for $param_name = $val...")

        # Construct new parameter set using baseline as template
        # We splat the baseline and overwrite the specific variable
        p_args = Dict(pairs(baseline_params))
        p_args[param_name] = val
        
        new_p = ModelParameters(
            n = N_FIRMS,
            σ = p_args[:σ],
            γ = p_args[:γ],
            β = p_args[:β],
            α = p_args[:α]
        )

        # Update structural part of the model
        update_model_params!(model, new_p)

        # Solve (Warm start: solve_PFI! uses the existing model.state)
        SchumpeterCournot.solve_PFI!(model)

        # Save result
        save_model(model, dir="saved_models/3-firm")
    end
    
    # Reset model to baseline before moving to the next parameter sweep
    # This ensures "one at a time" logic
    update_model_params!(model, ModelParameters(
        n = N_FIRMS, 
        σ = baseline_params.σ, 
        γ = baseline_params.γ, 
        β = baseline_params.β, 
        α = baseline_params.α
    ))
end

println("Batch processing complete.")