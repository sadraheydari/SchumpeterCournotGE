# =====================================================================
#  script/duopoly_sweep.jl
#
#  Extensive analysis of the n = 2 (duopoly) model. Each draw: sample a
#  calibration at random from a parameter domain, build+solve+simulate it
#  via `run_model`, save the eight-panel `plot_results` figure, dump the
#  printed report table to a .txt file, and append one row to a CSV.
#
#      julia --project script/duopoly_sweep.jl            # default budget
#      julia --project script/duopoly_sweep.jl 1000        # 1000 draws total
#
#  Durability: the CSV is written and flushed one row at a time (never held
#  in memory and dumped at the end), and a run that finds N existing rows on
#  disk resumes at draw N+1 — a killed or crashed run loses at most the row
#  in flight. No model is ever written to disk, only the CSV row, the PNG
#  and the report .txt for that draw.
#
#  A failed draw (VFI diverges, aggregate loop errors, ...) is still logged:
#  its row gets status = "error" and the exception message, and the sweep
#  moves on to the next draw rather than dying.
# =====================================================================

using Random, Printf, Dates, Statistics

include(joinpath(@__DIR__, "..", "src", "SchumpeterCournotGE.jl"))
using .SchumpeterCournotGE

# =====================================================================
#  1. The parameter domain
#
#  Ranges are uniform on the interval shown. `n = 2` and the grid are held
#  fixed across draws — only the economic primitives vary. Loosen or
#  tighten these to whatever region of the calibration space you actually
#  want covered; nothing else in the script assumes particular bounds.
# =====================================================================

const N = 2

const DOMAIN = (
    β = (0.94, 0.98),   # discount factor
    σ = (1.2,  4.0),    # inverse EIS
    μ = (2.0,  6.0),    # elasticity of substitution across industries
    γ = (1.005, 1.1),   # innovation step size
    θ = (0.0, 0.4),     # research elasticity  (optimal_research needs θ ≤ 1)
    ε = (0.0,  3.0),    # catch-up parameter
    η̄ = (0.1,  5.0),    # innovation scale
)

const AMIN           = 0.001
const K              = 100
const SPACING        = :log
const SPACING_PARAM  = 2.0

# --- starting point for the aggregate loop (same guess for every draw) --
const GW0   = 0.02
const GY0   = 0.02
const YHAT0 = 1.3

# --- loop tolerances / iteration budgets --------------------------------
const TOL_VFI      = 1e-10
const MAXITER_VFI  = 8_000
const TOL_SYM      = 1e-7
const MAXITER_SYM  = 150
const LAMBDA_SYM   = 0.25
const TOL_AGG      = 1e-5
const MAXITER_AGG  = 40
const LAMBDA_AGG   = 0.4

# --- simulation ----------------------------------------------------------
const N_SIMS    = 1_000
const N_PERIODS = 300
const BURNIN    = 100
const THIN      = 5

"""
    amax_for(μ) -> Float64

The grid's top end, sized to the draw: the stationary mean sits at
`μ/(μ-1)`, which blows up as `μ → 1`, so a fixed `amax` would starve the
low-μ draws of headroom while wasting nodes on the high-μ ones. Clamped so
neither end of the μ range produces a degenerate grid.
"""
amax_for(μ::Real) = clamp(8.0 * μ / (μ - 1.0), 6.0, 40.0)

"""
    sample_params(rng) -> NamedTuple

One calibration, drawn uniformly from [`DOMAIN`](@ref).
"""
function sample_params(rng::AbstractRNG)
    draw(bounds::Tuple) = bounds[1] + (bounds[2] - bounds[1]) * rand(rng)
    return NamedTuple{keys(DOMAIN)}(draw.(values(DOMAIN)))
end

# =====================================================================
#  2. CSV — fixed column order, written line by line
# =====================================================================

const COLUMNS = [
    "run_id", "seed", "timestamp", "status", "error",
    "n", "β", "σ", "μ", "γ", "θ", "ε", "η̄",
    "amin", "amax", "k", "spacing", "spacing_param",
    "agg_converged", "agg_iters", "agg_residual",
    "vfi_iters_last", "sym_iters_last", "outside_frac_last",
    "g_w", "g_y", "yhat", "L_r", "Lscript",
    "lerner_mean", "lerner_sd", "hhi_mean",
    "a_tilde_mean", "a_tilde_sd",
    "price_mean", "price_sd",
    "share_mean", "share_sd",
    "research_mean", "eta_mean",
    "rel_pos_ergodic_mean", "rel_pos_sym_mean",
    "gap_mean", "p_gap_gt_1_5", "p_firm_shutout",
    "elapsed_sec", "plot_path", "report_path",
]

"Render one value as a CSV field, quoting it if it contains a comma, quote or newline."
function csv_field(x)
    x === nothing && return ""
    s = x isa AbstractFloat && isnan(x) ? "" : string(x)
    occursin(r"[,\"\n]", s) || return s
    return "\"" * replace(s, "\"" => "\"\"") * "\""
end

function write_row(io::IO, row::Dict{String,Any})
    println(io, join((csv_field(get(row, c, "")) for c in COLUMNS), ","))
    flush(io)                                     # durable: survives a kill -9
end

# =====================================================================
#  3. One draw
# =====================================================================

"""
    run_one_draw(i, seed, figs_dir, reports_dir) -> Dict{String,Any}

Sample calibration `i`, solve it, and return its CSV row as a dict.
`i` alone determines the calibration (`rng = Xoshiro(seed + i)`), so draw
`i` is exactly reproducible regardless of where in a resumed run it falls.

Never throws: any failure — an infeasible aggregate loop, a divergent VFI,
... — is caught and recorded as `status = "error"` with the exception
message, so one bad draw cannot stop the sweep.
"""
function run_one_draw(i::Int, seed::Int, figs_dir::AbstractString,
                      reports_dir::AbstractString)
    t0  = time()
    rng = Random.Xoshiro(seed + i)
    p   = sample_params(rng)
    amax = amax_for(p.μ)

    row = Dict{String,Any}(
        "run_id" => i, "seed" => seed + i, "timestamp" => string(Dates.now()),
        "n" => N, "β" => p.β, "σ" => p.σ, "μ" => p.μ, "γ" => p.γ,
        "θ" => p.θ, "ε" => p.ε, "η̄" => p.η̄,
        "amin" => AMIN, "amax" => amax, "k" => K,
        "spacing" => String(SPACING), "spacing_param" => SPACING_PARAM,
    )

    tag = @sprintf("run_%05d_mu%.2f_g%.3f_th%.2f_eps%.2f_eta%.2f", i, p.μ, p.γ, p.θ, p.ε, p.η̄)
    plot_path   = joinpath(figs_dir, tag * ".png")
    report_path = joinpath(reports_dir, tag * ".txt")

    try
        res = run_model(; n = N, β = p.β, σ = p.σ, μ = p.μ, γ = p.γ,
                          θ = p.θ, ε = p.ε, η̄ = p.η̄,
                          amin = AMIN, amax = amax, k = K,
                          spacing = SPACING, spacing_param = SPACING_PARAM,
                          g_w = GW0, g_y = GY0, ŷ = YHAT0,
                          tol_vfi = TOL_VFI, maxiter_vfi = MAXITER_VFI,
                          tol_sym = TOL_SYM, maxiter_sym = MAXITER_SYM,
                          λ_sym = LAMBDA_SYM,
                          tol_agg = TOL_AGG, maxiter_agg = MAXITER_AGG,
                          λ_agg = LAMBDA_AGG,
                          n_sims = N_SIMS, n_periods = N_PERIODS, burnin = BURNIN,
                          seed = seed + i,
                          thin = THIN, progress = true, verbose = false)

        old_stdout = stdout
        rd, wr = redirect_stdout()
        try
            report(res)
        finally
            redirect_stdout(old_stdout)
            close(wr)
        end
        write(report_path, read(rd, String))

        plot_results(res; save = plot_path)

        st        = res.status
        agg       = res.model.sol.aggs
        sol       = res.model.sol
        pnl, spnl = res.panel, res.sym_panel
        tr        = isempty(res.trace) ? nothing : res.trace[end]

        merge!(row, Dict{String,Any}(
            "status" => "ok", "error" => "",
            "agg_converged" => st.converged, "agg_iters" => st.iters,
            "agg_residual"  => st.residual,
            "vfi_iters_last"     => tr === nothing ? "" : tr.vfi,
            "sym_iters_last"     => tr === nothing ? "" : tr.sym,
            "outside_frac_last"  => tr === nothing ? "" : tr.outside,
            "g_w" => agg.g_w, "g_y" => agg.g_y, "yhat" => agg.ŷ,
            "L_r" => sol.L_r, "Lscript" => sol.ℒ,
            "lerner_mean" => mean(pnl.lerner), "lerner_sd" => std(pnl.lerner),
            "hhi_mean" => p.μ * mean(pnl.lerner),
            "a_tilde_mean" => mean(pnl.a_tilde), "a_tilde_sd" => std(pnl.a_tilde),
            "price_mean" => mean(pnl.price), "price_sd" => std(pnl.price),
            "share_mean" => mean(pnl.share), "share_sd" => std(pnl.share),
            "research_mean" => mean(pnl.research), "eta_mean" => mean(pnl.eta),
            "rel_pos_ergodic_mean" => mean(pnl.rel_pos),
            "rel_pos_sym_mean" => mean(spnl.rel_pos),
            "gap_mean" => mean(spnl.gap),
            "p_gap_gt_1_5" => count(>(1.5), spnl.gap) / length(spnl.gap),
            "p_firm_shutout" => count(<(N), spnl.n_active) / length(spnl.n_active),
            "elapsed_sec" => time() - t0,
            "plot_path" => plot_path, "report_path" => report_path,
        ))
    catch e
        row["status"]      = "error"
        row["error"]       = sprint(showerror, e)
        row["elapsed_sec"] = time() - t0
        row["plot_path"]   = ""
        row["report_path"] = ""
    end
    return row
end

# =====================================================================
#  4. The sweep
# =====================================================================

"""
    run_sweep(; n_draws, seed, csv_path, figs_dir, reports_dir) -> String

Draw calibrations `1:n_draws`, solving and logging each in turn. Resumable:
if `csv_path` already holds `N` rows, draws `1:N` are skipped and the sweep
continues at `N+1`, so bumping `n_draws` and re-running extends a previous
sweep rather than restarting it.
"""
function run_sweep(; n_draws::Int = 300, seed::Int = 20260730,
                     csv_path::AbstractString = joinpath(@__DIR__, "..", "output", "n2-duopoly", "duopoly_sweep.csv"),
                     figs_dir::AbstractString = joinpath(@__DIR__, "..", "output", "n2-duopoly", "figs"),
                     reports_dir::AbstractString = joinpath(@__DIR__, "..", "output", "n2-duopoly", "reports"))
    mkpath(figs_dir)
    mkpath(reports_dir)
    mkpath(dirname(csv_path))

    start_idx = isfile(csv_path) ? max(0, countlines(csv_path) - 1) : 0
    if start_idx >= n_draws
        println("$csv_path already has $start_idx draws ≥ n_draws = $n_draws — nothing to do")
        return csv_path
    end
    start_idx > 0 && @printf("resuming %s at draw %d\n", csv_path, start_idx + 1)

    println("domain: ", DOMAIN)
    @printf("grid:   amin=%.3f  amax=amax(μ)∈[%.1f,%.1f]  k=%d  spacing=%s(%.2f)\n",
            AMIN, amax_for(DOMAIN.μ[2]), amax_for(DOMAIN.μ[1] + 1e-6), K,
            SPACING, SPACING_PARAM)

    io = open(csv_path, start_idx == 0 ? "w" : "a")
    if start_idx == 0
        println(io, join(COLUMNS, ","))
        flush(io)
    end

    try
        for i in (start_idx + 1):n_draws
            row = run_one_draw(i, seed, figs_dir, reports_dir)
            write_row(io, row)
            status_msg = row["status"] == "ok" ?
                @sprintf("g_w=%+.5f  ℒ=%.4f  L_r=%.4f", row["g_w"], row["Lscript"], row["L_r"]) :
                "ERROR: " * first(row["error"], 60)
            @printf("[%4d/%4d]  μ=%.2f γ=%.3f θ=%.2f ε=%.2f η̄=%.2f  |  %-46s  (%.1fs)\n",
                    i, n_draws, row["μ"], row["γ"], row["θ"], row["ε"], row["η̄"],
                    status_msg, row["elapsed_sec"])
        end
    finally
        close(io)
    end

    println("done — ", n_draws, " draws in ", csv_path)
    return csv_path
end

# =====================================================================
#  Run on load
# =====================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    n_draws = isempty(ARGS) ? 300 : parse(Int, ARGS[1])
    run_sweep(; n_draws = n_draws)
end

