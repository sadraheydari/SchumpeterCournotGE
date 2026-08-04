include("../src/SchumpeterCournotGE.jl")
using .SchumpeterCournotGE

res = run_model(
    n=4,
    β = 0.978, σ = 3.34, μ = 3.87, γ = 1.08,
    θ = 0.25, 
    ε = 2.76, 
    # ε = 0.75,
    η̄ = 1.35,
    amin = 0.001, amax = 12.5, k = 50, ky=10,
    spacing = :log, spacing_param = 2.0,
    g_w = 0.024902, g_y = 0.024902, ŷ = 1.139090, # n=3 res
    # g_w = 0.029027, g_y = 0.029027, ŷ = 1.050511, # n=4 res
    tol_vfi = 1e-9, maxiter_vfi = 1_000,
    tol_sym = 1e-7, maxiter_sym = 150, λ_sym = 0.1,
    tol_agg = 1e-5, maxiter_agg = 100, λ_agg = 0.1,
    n_sims = 1_000, n_periods = 500, burnin = 100,
    seed = 20260910
)

plot_results(res; save = "output/run_sample_n4.png")