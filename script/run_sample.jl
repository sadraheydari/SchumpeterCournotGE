include("../src/SchumpeterCournotGE.jl")
using .SchumpeterCournotGE

res = run_model(
    n=4,
    β = 0.96, σ = 2.0, μ = 2.0, γ = 1.06,
    θ = 0.20, ε = 1.50, η̄ = 1.5,
    amin = 0.05, amax = 6.0, k = 50, ky=20,
    spacing = :log, spacing_param = 2.0,
    g_w = 0.023552, g_y = 0.023552, ŷ = 1.202681,
    tol_vfi = 1e-9, maxiter_vfi = 1_000,
    tol_sym = 1e-7, maxiter_sym = 150, λ_sym = 0.25,
    tol_agg = 1e-5, maxiter_agg = 40, λ_agg = 0.4,
    n_sims = 3_000, n_periods = 1000, burnin = 300,
    seed = 20260730
)
plot_results(res; save = "output/run_sample.png")