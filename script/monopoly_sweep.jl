# =====================================================================
#  script/monopoly_sweep.jl
#
#  Extensive analysis of the n = 1 (monopoly) model. Each draw: sample a
#  calibration at random from a parameter domain, solve for g_w, record
#  moments of the stationary cross-section, save a four-panel plot, and
#  append one row to a CSV.
#
#      julia --project script/monopoly_sweep.jl            # default budget
#      julia --project script/monopoly_sweep.jl 1000        # 1000 draws total
#
#  Durability: the CSV is written and flushed one row at a time (never held
#  in memory and dumped at the end), and a run that finds N existing rows on
#  disk resumes at draw N+1 — a killed or crashed run loses at most the row
#  in flight. No model is ever written to disk, only the CSV row and the
#  PNG for that draw.
#
#  A failed draw (VFI diverges, no g_w brackets, ...) is still logged: its
#  row gets status = "error" and the exception message, and the sweep moves
#  on to the next draw rather than dying.
# =====================================================================

using Random, Printf, Dates

include(joinpath(@__DIR__, "..", "src", "SchumpeterCournotGE.jl"))

# =====================================================================
#  1. The parameter domain
#
#  Ranges are uniform on the interval shown. `n`, `amin`, the grid size and
#  spacing are held fixed across draws — only the economic primitives vary.
#  Loosen or tighten these to whatever region of the calibration space you
#  actually want covered; nothing else in the script assumes particular
#  bounds.
# =====================================================================

const DOMAIN = (
    β = (0.94, 0.98),   # discount factor
    σ = (1.2,  4.0),    # inverse EIS
    μ = (2.0,  6.0),    # elasticity of substitution across industries
    γ = (1.005, 1.1),   # innovation step size
    θ = (0.0, 0.75),   # research elasticity  (optimal_research needs θ ≤ 1)
    ε = (0.0,  3.0),    # catch-up parameter
    η̄ = (0.1,  5.0),    # innovation scale
)

const AMIN           = 0.05
const KX              = 200
const SPACING         = :power
const SPACING_PARAM   = 1.5

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
#  2. Moments of the stationary cross-section
# =====================================================================

"""
    moments(a, φ) -> NamedTuple

The first four (central, for 2–4) moments of `a` under the stationary
distribution `φ`: mean, variance, standard deviation, skewness, kurtosis
(raw, not excess — 3.0 is the normal benchmark).
"""
function moments(a::AbstractVector{<:Real}, φ::AbstractVector{<:Real})
    m1 = sum(φ .* a)
    c2 = sum(φ .* (a .- m1) .^ 2)
    c3 = sum(φ .* (a .- m1) .^ 3)
    c4 = sum(φ .* (a .- m1) .^ 4)
    sd = sqrt(c2)
    return (mean = m1, var = c2, sd = sd,
            skewness = sd > 0 ? c3 / sd^3 : NaN,
            kurtosis = sd > 0 ? c4 / sd^4 : NaN)
end

# =====================================================================
#  3. CSV — fixed column order, written line by line
# =====================================================================

const COLUMNS = [
    "run_id", "seed", "timestamp", "status", "error",
    "n", "β", "σ", "μ", "γ", "θ", "ε", "η̄",
    "amin", "amax", "kx", "spacing", "spacing_param",
    "vfi_converged", "vfi_iters", "vfi_residual",
    "gw_converged", "gw_residual", "feasible_floor",
    "g_w", "g_y", "yhat_given", "yhat_implied",
    "level", "L_r", "Lscript", "eta_mean",
    "dist_iters", "dist_gap", "mass_lo", "mass_hi", "boundary_flag",
    "a_mean", "a_var", "a_sd", "a_skew", "a_kurt",
    "elapsed_sec", "plot_path",
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
#  4. One draw
# =====================================================================

"""
    run_one_draw(i, seed, figs_dir) -> Dict{String,Any}

Sample calibration `i`, solve it, and return its CSV row as a dict.
`i` alone determines the calibration (`rng = Xoshiro(seed + i)`), so draw
`i` is exactly reproducible regardless of where in a resumed run it falls.

Never throws: any failure — an infeasible growth floor, a bracket that
never flips sign, a divergent VFI — is caught and recorded as
`status = "error"` with the exception message, so one bad draw cannot stop
the sweep.
"""
function run_one_draw(i::Int, seed::Int, figs_dir::AbstractString)
    t0  = time()
    rng = Random.Xoshiro(seed + i)
    p   = sample_params(rng)
    amax = amax_for(p.μ)

    row = Dict{String,Any}(
        "run_id" => i, "seed" => seed + i, "timestamp" => string(Dates.now()),
        "n" => 1, "β" => p.β, "σ" => p.σ, "μ" => p.μ, "γ" => p.γ,
        "θ" => p.θ, "ε" => p.ε, "η̄" => p.η̄,
        "amin" => AMIN, "amax" => amax, "kx" => KX,
        "spacing" => String(SPACING), "spacing_param" => SPACING_PARAM,
    )

    try
        model = build_monopoly(; β = p.β, σ = p.σ, μ = p.μ, γ = p.γ,
                                  θ = p.θ, ε = p.ε, η̄ = p.η̄,
                                  amin = AMIN, amax = amax, kx = KX,
                                  spacing = SPACING, spacing_param = SPACING_PARAM)
        ws = VFIWorkspace(model)
        gw = find_g_w!(model, ws; verbose = false)
        vfi_st = model.sol.vfi          # state of the VFI at the converged g

        φ, dit, dgap = stationary_distribution(model)
        imp          = implied_aggregates(model, φ)
        a            = nodes(xaxis(model.grid))
        mom          = moments(a, φ)
        lo_m, hi_m   = boundary_mass(φ)

        plot_path = joinpath(figs_dir,
            @sprintf("run_%05d_mu%.2f_g%.3f_th%.2f_eps%.2f_eta%.2f.png",
                     i, p.μ, p.γ, p.θ, p.ε, p.η̄))
        res = (model = model, ws = ws, status = vfi_st, growth = gw,
               φ = φ, implied = imp)
        plot_monopoly(res; save = plot_path)

        merge!(row, Dict{String,Any}(
            "status" => "ok", "error" => "",
            "vfi_converged" => vfi_st.converged, "vfi_iters" => vfi_st.iters,
            "vfi_residual"  => vfi_st.residual,
            "gw_converged"  => gw.converged, "gw_residual" => gw.residual,
            "feasible_floor" => feasible_growth_floor(model.params),
            "g_w" => model.sol.aggs.g_w, "g_y" => model.sol.aggs.g_y,
            "yhat_given" => model.sol.aggs.ŷ, "yhat_implied" => imp.ŷ,
            "level" => imp.level, "L_r" => imp.L_r, "Lscript" => imp.ℒ,
            "eta_mean" => imp.η_mean,
            "dist_iters" => dit, "dist_gap" => dgap,
            "mass_lo" => lo_m, "mass_hi" => hi_m,
            "boundary_flag" => max(lo_m, hi_m) > 1e-3,
            "a_mean" => mom.mean, "a_var" => mom.var, "a_sd" => mom.sd,
            "a_skew" => mom.skewness, "a_kurt" => mom.kurtosis,
            "elapsed_sec" => time() - t0, "plot_path" => plot_path,
        ))
    catch e
        row["status"]      = "error"
        row["error"]       = sprint(showerror, e)
        row["elapsed_sec"] = time() - t0
        row["plot_path"]   = ""
    end
    return row
end

# =====================================================================
#  5. The sweep
# =====================================================================

"""
    run_sweep(; n_draws, seed, csv_path, figs_dir) -> String

Draw calibrations `1:n_draws`, solving and logging each in turn. Resumable:
if `csv_path` already holds `N` rows, draws `1:N` are skipped and the sweep
continues at `N+1`, so bumping `n_draws` and re-running extends a previous
sweep rather than restarting it.
"""
function run_sweep(; n_draws::Int = 300, seed::Int = 20260729,
                     csv_path::AbstractString = joinpath(@__DIR__, "..", "output", "n1-monopoly", "monopoly_sweep.csv"),
                     figs_dir::AbstractString = joinpath(@__DIR__, "..", "output", "n1-monopoly", "figs"))
    mkpath(figs_dir)
    mkpath(dirname(csv_path))

    start_idx = isfile(csv_path) ? max(0, countlines(csv_path) - 1) : 0
    if start_idx >= n_draws
        println("$csv_path already has $start_idx draws ≥ n_draws = $n_draws — nothing to do")
        return csv_path
    end
    start_idx > 0 && @printf("resuming %s at draw %d\n", csv_path, start_idx + 1)

    println("domain: ", DOMAIN)
    @printf("grid:   amin=%.3f  amax=amax(μ)∈[%.1f,%.1f]  kx=%d  spacing=%s(%.2f)\n",
            AMIN, amax_for(DOMAIN.μ[2]), amax_for(DOMAIN.μ[1] + 1e-6), KX,
            SPACING, SPACING_PARAM)

    io = open(csv_path, start_idx == 0 ? "w" : "a")
    if start_idx == 0
        println(io, join(COLUMNS, ","))
        flush(io)
    end

    try
        for i in (start_idx + 1):n_draws
            row = run_one_draw(i, seed, figs_dir)
            write_row(io, row)
            status_msg = row["status"] == "ok" ?
                @sprintf("g_w=%+.5f  level=%.4f", row["g_w"], row["level"]) :
                "ERROR: " * first(row["error"], 60)
            @printf("[%4d/%4d]  μ=%.2f γ=%.3f θ=%.2f ε=%.2f η̄=%.2f  |  %-38s  (%.1fs)\n",
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
