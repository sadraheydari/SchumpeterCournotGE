include("../src/SchumpeterCournotGE.jl")
using .SchumpeterCournotGE

res = run_model(
    n=3,
    β = 0.96, σ = 2.0, μ = 2.0, γ = 1.06,
    θ = 0.10, ε = 1.50, η̄ = 1.5,
    amin = 0.05, amax = 8.0, k = 50,
    spacing = :log, spacing_param = 2.0,
    g_w = 0.021526, g_y = 0.021526, ŷ = 1.334950,
    tol_vfi = 1e-10, maxiter_vfi = 8_000,
    tol_sym = 1e-7, maxiter_sym = 150, λ_sym = 0.25,
    tol_agg = 1e-5, maxiter_agg = 40, λ_agg = 0.4,
    n_sims = 2_000, n_periods = 500, burnin = 100,
    seed = 20260730
)