function policy_by_relative_state(model::DSCIModel; debug::Bool=false)

    env = model.env
    state = model.state

    n = env.param.n
    τ_max = env.τ_max
    idx_map = env.idx_map

    dim = n - 1

    # Difference states range
    d_states = collect(-τ_max:τ_max)
    len_d = length(d_states)

    # Create (n-1)-dimensional array
    dims = ntuple(_ -> len_d, dim)
    p_grid = fill(NaN, dims...)

    # Precompute index lookup for speed
    tuple_to_idx = idx_map.tuple_to_idx

    # Cartesian loop over all d combinations
    @threads for I in CartesianIndices(p_grid)

        # Extract d vector
        d_vec = [d_states[i] for i in Tuple(I)]

        # Enforsce sorted state constraint (A_vec must be sorted)
        # Equivalent to your `if d_i > d_j continue`
        if !issorted(d_vec)
            continue
        end

        debug && println("Processing d_vec = ", d_vec)

        p_vals = Float64[]
        sizehint!(p_vals, τ_max)

        for A in 1:τ_max

            # Recover original state: A_vec = d + A + 1
            state_vec = d_vec .+ A .+ 1

            try
                idx = tuple_to_idx[state_vec]
                push!(p_vals, state.policy_grid[A, idx])
                debug && println("  A=$A → policy=", p_vals[end])
            catch
                debug && println("  State not found: ", state_vec)
                continue
            end
        end

        debug && println("  Collected policies: ", p_vals)

        if !isempty(p_vals)
            p_val = median(p_vals)
            idx_tuple = Tuple(I)
            debug && println("  median=", p_val)
            debug && println("  Assigning to indices: ", unique(permutations(idx_tuple)))
            for perm in unique(permutations(idx_tuple))
                debug && println("    Assigning p_grid[", perm, "] = ", p_val) 
                p_grid[perm...] = p_val
            end
            debug && println("  median=", p_val)
        end
    end

    return p_grid, d_states
end;
