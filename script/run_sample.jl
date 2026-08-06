include("../src/SchumpeterCournotGE.jl")
using .SchumpeterCournotGE

res = run_model(
           n=3,
           β = 0.97, σ = 1.5, μ = 3.5, γ = 1.05,
           θ = 0.1, ε = 1.5, η̄ = 0.7,
           amin = 0.05, amax = 5.0, k = 100,
           spacing = :log, spacing_param = 10.0,
           g_w = 0.01494, 
           g_y = 0.01494, 
           ŷ = 1.2012,
           tol_vfi = 1e-9, maxiter_vfi = 1_000,
           tol_sym = 1e-7, maxiter_sym = 150, λ_sym = 0.3,
           tol_agg = 1e-5, maxiter_agg = 100, λ_agg = 0.2,
           n_sims = 5_000, n_periods = 1_300, burnin = 300,
           seed = 20260910
       )

save_config("output/n3-triopoly/models/callibrated.toml", res.model)
save_model("output/n3-triopoly/models/callibrated.jld2", res.model)

set = SimSettings(n_industries = 5_000,
                  n_periods    = 1_300,
                  burnin       = 300,
                  thin         = 1,
                  seed         = 20260801)

sim = run_simulation(res.model, set)
sim_report(sim)
sim_report(sim;        save = "output/n3-triopoly/sim")
plot_stationarity(sim; save = "output/n3-triopoly/sim")
plot_industry(sim;     save = "output/n3-triopoly/sim")