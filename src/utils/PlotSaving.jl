function model_to_dict(model::DSCIModel)

    return Dict(
        "Timestamp" => string(Dates.now()),

        "ModelParameters" => Dict(
            "n" => model.env.param.n,
            "sigma" => model.env.param.σ,
            "beta" => model.env.param.β,
            "gamma" => model.env.param.γ,
            "alpha" => model.env.param.α,
            "PROB" => model.env.param.PROB
        ),

        "ModelEnvironment" => Dict(
            "tau_max" => model.env.τ_max,
            "l_max" => model.env.l_max
        ),

        "ModelSettings" => Dict(
            "max_iter_update" => model.settings.max_iter_update,
            "tol_update" => model.settings.tol_update,
            "clamp_rate_update" => model.settings.clamp_rate_update,
            "sdf_relaxer" => model.settings.sdf_relaxer
        ),

        "SolverState" => Dict(
            "is_converged" => model.state.is_converged
        )
    )
end;



function build_footer_strings(model::DSCIModel)

    # -----------------------------
    # Economic parameters
    # -----------------------------
    econ = [
        "n=$(model.env.param.n)",
        @sprintf("σ=%.3g", model.env.param.σ),
        @sprintf("β=%.3g", model.env.param.β),
        @sprintf("γ=%.3g", model.env.param.γ),
        @sprintf("α=%.3g", model.env.param.α),
        "PROB=$(model.env.param.PROB)"
    ]

    econ_line = join(econ, "   ")

    # -----------------------------
    # Algorithmic parameters
    # -----------------------------
    algo = [
        "τ=$(model.env.τ_max)",
        @sprintf("l_max=%.3g", model.env.l_max),
        @sprintf("tol=%.1e", model.settings.tol_update),
        @sprintf("sdf=%.3g", model.settings.sdf_relaxer)
    ]

    algo_line = join(algo, "   ")

    return econ_line, algo_line
end;




function build_footer_plot(model::DSCIModel; fsize=7)

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


function combine_with_footer(plt, model::DSCIModel; fsize=7)

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
    model::DSCIModel,
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
