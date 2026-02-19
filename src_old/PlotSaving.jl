using JSON3
using Dates
using Plots

function model_to_dict(model::ModelSettings)

    return Dict(
        "Timestamp" => string(Dates.now()),

        "ModelParameters" => Dict(
            "n" => model.param.n,
            "sigma" => model.param.σ,
            "beta" => model.param.β,
            "gamma" => model.param.γ,
            "alpha" => model.param.α,
            "PROB" => model.param.PROB
        ),

        "AlgorithmicParameters" => Dict(
            "tau_max" => model.τ_max,
            "l_max" => model.l_max,
            "tol_PF_update" => model.tol_PF_update,
            "tol_PF_solver" => model.tol_PF_solver,
            "max_iter_PF_solver" => model.max_iter_PF_solver,
            "max_iter_PF_update" => model.max_iter_PF_update,
            "clamp_rate_PF_solver" => model.clamp_rate_PF_solver,
            "clamp_rate_PF_update" => model.clamp_rate_PF_update,
            "sdf_relaxer" => model.sdf_relaxer
        )
    )
end;



function build_footer_strings(model::ModelSettings)

    # -----------------------------
    # Economic parameters
    # -----------------------------
    econ = [
        "n=$(model.param.n)",
        @sprintf("σ=%.3g", model.param.σ),
        @sprintf("β=%.3g", model.param.β),
        @sprintf("γ=%.3g", model.param.γ),
        @sprintf("α=%.3g", model.param.α),
        "PROB=$(model.param.PROB)"
    ]

    econ_line = join(econ, "   ")

    # -----------------------------
    # Algorithmic parameters
    # -----------------------------
    algo = [
        "τ=$(model.τ_max)",
        @sprintf("l_max=%.3g", model.l_max),
        @sprintf("tol=%.1e", model.tol_PF_update),
        @sprintf("sdf=%.3g", model.sdf_relaxer)
    ]

    algo_line = join(algo, "   ")

    return econ_line, algo_line
end;




function build_footer_plot(model::ModelSettings; fsize=7)

    econ_line, algo_line = build_footer_strings(model)

    footer_text = econ_line * "\n" * algo_line

    footer = plot(
        framestyle = :none,
        xticks = false,
        yticks = false,
        legend = false,
        grid = false
    )

    annotate!(
        footer,
        0.5, 0.5,
        text(footer_text, fsize, :gray),
        :center
    )

    return footer
end;


function combine_with_footer(plt, model::ModelSettings; fsize=7)

    footer = build_footer_plot(model; fsize=fsize)

    combined = plot(
        plt,
        footer,
        layout = @layout([a; b{0.1h}])
    )

    return combined
end;


function build_clean_filename(
    plot_name::AbstractString;
    ext::AbstractString = "pdf"
)
    date_str = Dates.format(now(), "yyyy-mm-dd")
    time_str = Dates.format(now(), "HH-MM-SS")

    return "$plot_name [$date_str] ($time_str).$ext"
end;


function save_plot(
    plt,
    model::ModelSettings,
    path::AbstractString,
    plot_name::AbstractString;
    ext::AbstractString = "pdf",
    annotate_info::Bool = true,
    save_metadata::Bool = true,
    footer_fontsize::Int = 7
)

    isdir(path) || mkpath(path)

    filename = build_clean_filename(plot_name; ext=ext)
    fullpath = joinpath(path, filename)

    # Copy plot
    plt_to_save = deepcopy(plt)

    if annotate_info
        plt_to_save = combine_with_footer(
            plt_to_save,
            model;
            fsize=footer_fontsize
        )
    end

    savefig(plt_to_save, fullpath)

    if save_metadata
        metadata_dict = model_to_dict(model)
        json_path = replace(fullpath, ".$ext" => ".json")

        open(json_path, "w") do io
            JSON3.pretty(io, metadata_dict)
        end
    end

    return fullpath
end
