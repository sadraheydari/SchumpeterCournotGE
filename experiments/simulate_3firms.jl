using Pkg; Pkg.activate(".");

using SchumpeterCournot
using LinearAlgebra, Statistics, Plots
using .Threads
Plots.pyplot()


model = SchumpeterCournot.load_model("saved_models/3-firm/DSCI_2026-02-26_15-45-30.jld2")

# Simulate the model
function simulate_economy(model:: DSCIModel, A0:: Vector{Int}; timesteps:: Int, econ_counts:: Int)
    n_firms = model.env.param.n
    @assert length(A0) == n_firms "Initial action vector A0 must have length equal to number of firms"

    agent_states = Dict{String, Matrix{Int}}()
    agent_actions = Dict{String, Matrix{Float64}}()
    for i in 1:n_firms
        agent_states["firm_$i"] = zeros(econ_counts, timesteps)
        agent_states["firm_$i"][:, 1] .= A0[i]
        agent_actions["firm_$i"] = zeros(econ_counts, timesteps)
    end

    for t in 1:timesteps-1
        for econ in 1:econ_counts
            A_t = [agent_states["firm_$i"][econ, t] for i in 1:n_firms]
            actions = SchumpeterCournot.get_labour_demand(A_t, model)
            success_probs = [SchumpeterCournot.innovation_success_prob(a, model.env.param) for a in actions]

            for i in 1:n_firms
                agent_actions["firm_$i"][econ, t] = actions[i]
                # Simulate innovation success
                if rand() < success_probs[i]
                    agent_states["firm_$i"][econ, t+1] = A_t[i] + 1
                else
                    agent_states["firm_$i"][econ, t+1] = A_t[i]
                end
            end
        end
    end

    return agent_states, agent_actions 
end


# Plot the results
function plot_single_firm_simulation!(p, firm_id:: Int, agent_states:: Dict; color=:blue, econ_plot_step:: Int=20)
    firm_key = "firm_$firm_id"
    A = agent_states[firm_key]
    for econ in 1:econ_plot_step:size(A, 1)
        plot!(p, A[econ, :], label="", color=color, alpha=0.1)
    end
    return p
end


agent_states, agent_actions = simulate_economy(
    model, [7, 5, 1]; 
    timesteps=100, econ_counts=500
)

p_list = []
cmap = cgrad(:turbo, model.env.param.n);
for firm_id in 1:model.env.param.n
    initial_A = agent_states["firm_$firm_id"][1, 1]
    title_str = "Firm $firm_id (τ₀=$initial_A)"
    p = plot(xlabel="Time", ylabel="Innovation Level", title=title_str)
    ratio = float(firm_id) / model.env.param.n
    plot_single_firm_simulation!(p, firm_id, agent_states; color=cmap[ratio], econ_plot_step=20)
    push!(p_list, p)
end

# add mean innovation level across all economies
for firm_id in 1:model.env.param.n, i_p in 1:length(p_list)
    p = p_list[i_p]
    firm_key = "firm_$firm_id"
    A = agent_states[firm_key]
    mean_A = mean(A, dims=1)
    ratio = float(firm_id) / model.env.param.n
    alpha = i_p == firm_id ? 1.0 : 0.5
    style = i_p == firm_id ? :solid : :dash
    plot!(p, vec(mean_A), label="Firm $firm_id", color=cmap[ratio], linewidth=2, alpha=alpha, linestyle=style)
end

y_max = maximum([ylims(p)[2] for p in p_list])
for p in p_list
    ylims!(p, (0, y_max))
end

p = plot(
    p_list..., 
    size=(model.env.param.n * 400, 300),
    layout=(1, model.env.param.n), 
    legend=:topleft
)



save_plot(p, model, "output/3-firm", "innovation_simulation"; ext = "png")


p_list = []
cmap = cgrad(:turbo, model.env.param.n);
for firm_id in 1:model.env.param.n
    initial_A = agent_states["firm_$firm_id"][1, 1]
    title_str = "Firm $firm_id (τ₀=$initial_A)"
    p = plot(xlabel="Time", ylabel="Innovation Level", title=title_str)
    ratio = float(firm_id) / model.env.param.n
    plot_single_firm_simulation!(p, firm_id, agent_actions; color=cmap[ratio], econ_plot_step=20)
    push!(p_list, p)
end

# add mean innovation level across all economies
for firm_id in 1:model.env.param.n, i_p in 1:length(p_list)
    p = p_list[i_p]
    firm_key = "firm_$firm_id"
    A = agent_actions[firm_key]
    mean_A = mean(A, dims=1)
    ratio = float(firm_id) / model.env.param.n
    alpha = i_p == firm_id ? 1.0 : 0.5
    style = i_p == firm_id ? :solid : :dash
    plot!(p, vec(mean_A), label="Firm $firm_id", color=cmap[ratio], linewidth=2, alpha=alpha, linestyle=style)
end

y_max = maximum([ylims(p)[2] for p in p_list])
for p in p_list
    ylims!(p, (0, y_max))
end

p = plot(
    p_list..., 
    size=(model.env.param.n * 400, 300),
    layout=(1, model.env.param.n), 
    legend=:topleft
)

