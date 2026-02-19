function save_model(model::DSCIModel;
                    dir::AbstractString = "saved_models",
                    prefix::AbstractString = "DSCI",
                    save_metadata::Bool = true)

    isdir(dir) || mkpath(dir)

    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    base_name = "$(prefix)_$(timestamp)"
    jld_path = joinpath(dir, base_name * ".jld2")

    # Extract components
    param = model.env.param
    τ_max = model.env.τ_max
    l_max = model.env.l_max
    settings = model.settings
    V = model.state.V_grid
    P = model.state.policy_grid
    Pj = model.state.policy_grid_j

    # Save binary model
    @save jld_path param τ_max l_max settings V P Pj

    # Save metadata JSON
    if save_metadata
        metadata_dict = model_to_dict(model)
        metadata_dict["SavedAt"] = string(Dates.now())

        json_path = joinpath(dir, base_name * ".json")

        open(json_path, "w") do io
            JSON3.pretty(io, metadata_dict)
        end
    end

    return jld_path
end




function load_model(path::AbstractString)::DSCIModel

    @load path param τ_max l_max settings V P Pj

    idx_map = build_lookup(param.n - 1, τ_max)
    env = ModelEnvironment(param, τ_max, idx_map, l_max)

    state = SolverState(V, P, Pj)

    return DSCIModel(env, settings, state)
end

