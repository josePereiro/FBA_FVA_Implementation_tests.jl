# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: jl,ipynb
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Julia 1.1.0
#     language: julia
#     name: julia-1.1
# ---

import Plots
Plots.pyplot();
import Serialization: serialize, deserialize
import JuMP
import GLPK
import Convex
import Clp
import MathProgBase
import JSON
using SparseArrays
using Test

# ---
# # Tools
# ---

function load_model(file)
    model_dict = JSON.parsefile(file)
    
    # Fix types
    model_dict["S"] = model_dict["S"] .|> Vector{Float64}
    model_dict["S"] = sparse(model_dict["S"]...) |> Matrix
    for k in ["b", "lb", "ub"]
        model_dict[k] = model_dict[k] |> Vector{Float64}
    end
    for k in ["rxns", "mets"]
        model_dict[k] = model_dict[k] |> Vector{String}
    end
    model_dict
end

rxnindex(model, ider::Int) = 1 <= ider <= size(model["S"], 2) ? ider : nothing
rxnindex(model, ider::AbstractString) = findfirst(isequal(ider), model["rxns"])

# ---
# # FVA
# ---

# For compute $lval_i$ (lowest value):
# $$ 
#     \text{minimize   $x_i$}\\
#     \text{subject to: }\\
#     S * x = b\\
#     lb \leq x \leq ub
# $$
#
# For compute $uval_i$ (uppest value):
# $$ 
#     \text{maximize   $x_i$}\\
#     \text{subject to: }\\
#     S * x = b\\
#     lb \leq x \leq ub
# $$

# ### JuMP

# +
function fva_JuMP(S, b, lb, ub, idxs = eachindex(lb); 
        verbose = true, 
        upfrec = 10, 
        zeroth = 1e-10,
        solver = Clp.Optimizer)
    
    verbose && (println("Creating LP model"); flush(stdout))    
    lp_model = JuMP.Model(solver)
    JuMP.set_optimizer_attribute(lp_model, "LogLevel", 0)

    verbose && (println("Creating LP variables"); flush(stdout))
    JuMP.@variable(lp_model, lp_x[1:size(S, 2)])
    
    verbose && (println("Creating LP contraints"); flush(stdout))
    JuMP.@constraint(lp_model, balance, S * lp_x .== b)
    JuMP.@constraint(lp_model, bounds, lb .<= lp_x .<= ub)

    lvals = zeros(eltype(S), length(idxs))
    uvals = zeros(eltype(S), length(idxs))
    
    verbose && (println("FVA Processing"); flush(stdout))
    n = length(idxs)
    upfrec = floor(Int, max(n/ upfrec, 1))
    
    for (i, idx) in enumerate(idxs)
        
        show_progress = verbose && (i == 1 || i == n || i % upfrec == 0)
        show_progress && (println("Doing $i / $n"); flush(stdout))
        
        # lower val
        JuMP.@objective(lp_model, JuMP.MOI.MIN_SENSE, lp_x[idx])
        JuMP.optimize!(lp_model)
        x = JuMP.value(lp_x[idx])
        lvals[i] = abs(x) < zeroth ? zero(x) : x
        
        # upper val
        JuMP.@objective(lp_model, JuMP.MOI.MAX_SENSE, lp_x[idx])
        JuMP.optimize!(lp_model)
        x = JuMP.value(lp_x[idx])
        uvals[i] = abs(x) < zeroth ? zero(x) : x
        
    end
    return (lvals, uvals)
end

function fva_JuMP(model, iders = eachindex(model["lb"]); kwargs...) 
    idxs = [rxnindex(model, idx) for idx in iders]
    fva_JuMP(model["S"], model["b"], model["lb"], model["ub"], idxs; kwargs...)
end
# -

# ### MathProgBase

# +
function fva_MathProgBase(S, b, lb, ub, idxs = eachindex(lb); 
        verbose = true, 
        upfrec = 10, 
        zeroth = 1e-10,
        solver = Clp.ClpSolver()
    )

    lvals = zeros(eltype(S), length(idxs))
    uvals = zeros(eltype(S), length(idxs))
    
    sv = zeros(size(S, 2));
    
    n = length(idxs)
    upfrec = floor(Int, max(n/ upfrec, 1))
    for (col, sense) in [(lvals, 1.0), (uvals, -1.0)]
        
        for (i, idx) in enumerate(idxs)

            show_progress = verbose && sense == 1 && (i == 1 || i % upfrec == 0 || i == n)
            show_progress && (println("fva[$i / $n]"); flush(stdout))

            sv[idx] = sense
            sol = MathProgBase.HighLevelInterface.linprog(
                sv, # Opt sense vector 
                S, # Stoichiometric matrix
                b, # row lb
                b, # row ub
                lb, # column lb
                ub, # column ub
                solver);
            isempty(sol.sol) && error("FBA failed, empty solution returned!!!")
            
            x = sol.sol[idx]
            col[i] = abs(x) < zeroth ? zero(x) : x
            sv[idx] = zero(sense)
        end
    end
    return (lvals, uvals)
end

function fva_MathProgBase(model, iders = eachindex(model["lb"]); kwargs...) 
    idxs = [rxnindex(model, idx) for idx in iders]
    return fva_MathProgBase(model["S"], model["b"], model["lb"], model["ub"], idxs; kwargs...);
end

fva_MathProgBase(model, ider::AbstractString; kwargs...) = fva_MathProgBase(model, [ider]; kwargs...)
# -

# ---
# # FBA
# ---

# ### Convex

# +
function fba_Convex(S, b, lb, ub, obj_idx; solver = GLPK.Optimizer)
    
    M, N = size(S)
    x = Convex.Variable(N)
    p = Convex.maximize(x[obj_idx]) 
    p.constraints += S * x == b
    p.constraints += x >= lb
    p.constraints += x <= ub
    Convex.solve!(p, solver)
    
    v = first(p.objective.children).value |> vec
    return (sol = v, obj_val = v[obj_idx], obj_idx = obj_idx)
end

function fba_Convex(model)
    obj_idx = rxnindex(model, model["obj_ider"])
    return fba_Convex(model["S"], model["b"], model["lb"], model["ub"], obj_idx)
end
# -

# ### JuMP

# +
function fba_JuMP(S, b, lb, ub, obj_idx::Integer; 
        sense = JuMP.MOI.MAX_SENSE, err = [], 
        lp_model = nothing,
        solver = GLPK.Optimizer)
    
    if isnothing(lp_model) 
        lp_model = JuMP.Model(solver)
    end

    M,N = size(S)

    # Variables
    JuMP.@variable(lp_model, lp_x[1:N])

    # Constraints
    JuMP.@constraint(lp_model, balance, S * lp_x .== b)
    JuMP.@constraint(lp_model, bounds, lb .<= lp_x .<= ub)

    # objective
    JuMP.@objective(lp_model, sense, lp_x[obj_idx])

    # optimize
    JuMP.optimize!(lp_model)
    
    #FBAout
    obj_val = JuMP.value(lp_x[obj_idx])
    obj_val in err && error("FBA failed, error value returned, obj_val[$(obj_idx)] = $(obj_val)!!!")
    
    return (sol = JuMP.value.(lp_x), obj_val = obj_val, obj_idx = obj_idx)
end

function fba_JuMP(model; kwargs...) 
    obj_idx = rxnindex(model, model["obj_ider"])
    return fba_JuMP(model["S"], model["b"], model["lb"], model["ub"], obj_idx; kwargs...)
end
# -

# ### MathProgBase

# +
function fba_MathProgBase(S, b, lb, ub, obj_idx::Integer; 
        sense = -1.0, solver = Clp.ClpSolver())
    sv = zeros(size(S, 2));
    sv[obj_idx] = sense
    sol = MathProgBase.HighLevelInterface.linprog(
        sv, # Opt sense vector 
        S, # Stoichiometric matrix
        b, # row lb
        b, # row ub
        lb, # column lb
        ub, # column ub
        solver);
    isempty(sol.sol) && error("FBA failed, empty solution returned!!!")
    return (sol = sol.sol, obj_val = sol.sol[obj_idx], obj_idx = obj_idx)
end

function fba_MathProgBase(model)
    obj_idx = rxnindex(model, model["obj_ider"])
    return fba_MathProgBase(model["S"], model["b"], model["lb"], model["ub"], obj_idx)
end
# -

# ---
# # Tests
# ---

model_files = ["toy_model.json", "iJR904.json", "HumanGEM.json"][1:2]
@assert all(isfile.(model_files))

# ### Testing FBA

function test_fba(tname, fba_fun, model_file)
    println("Test: ", tname, " --------------- ")
    println("Model: ", basename(model_file))
    model = load_model(model_file);
    @time fbaout = fba_fun(model); flush(stdout)
    println("obj_val: ", fbaout.obj_val)
    println("flux sum: ", sum(fbaout.sol))
    println("error sum: ", sum(abs.(model["S"] * fbaout.sol - model["b"])))
    println()
    flush(stdout)
    return fbaout
end

fba_tests = Dict()
for model_file in model_files
    fba_tests[model_file] = Dict()
    println("\n ------------- $model_file -------------")
    for (tname, fba_fun) in [("FBA JuMP-GLPK", fba_JuMP), 
                            ("FBA Convex-GLPK", fba_Convex),
                            ("FBA MathProgBase-Clp", fba_MathProgBase)]
        fba_tests[model_file][tname] = test_fba(tname, fba_fun, model_file)
    end
end

for (model_file, results) in fba_tests
    @testset "$model_file" begin
        obj_vals = [fbaout.obj_val for (tname, fbaout) in results]
        # all fba solutions must be the same
        @test all(isapprox.(obj_vals |> first, obj_vals))
    end
end

ps = []
for (model_file, results) in fba_tests
    model_ps = []
    for (tbame, fbaout) in results
        p = Plots.plot(title = "$(basename(model_file))\n$(tbame)", xlabel = "flux idx", ylabel = "flux value")
        Plots.plot!(fbaout.sol, label = "")
        push!(model_ps, p)
    end
    p = Plots.plot(model_ps..., size = (900, 300), layout = (1,3))
    push!(ps, p)
end
p = Plots.plot(ps..., size = (900, 800), layout = (3,1), titlefont = 10)

# ### Testing FVA preprocess model

function fva_pp_model_test(tname, fva_method, fba_method, model_file)
    println("Test: ", tname, " --------------- ")
    println("Model: ", basename(model_file))
    model = load_model(model_file);
    obj_idx = rxnindex(model, model["obj_ider"])
    
    println("\nOriginal model FBA")
    orig_fbaout = fba_method(model)
    println("obj_val: ", orig_fbaout.obj_val)
    println("flux sum: ", sum(orig_fbaout.sol))
    println("error sum: ", sum(abs.(model["S"] * orig_fbaout.sol - model["b"])))
    
    println("\nDoing FVA")
    @time lvals, uvals = fva_method(model);
    flush(stdout)
    # Here I check that the fba performed in the fva method returns th same solutions
    # as the fba implementation in the fba method
    println("uvals[obj_idx]: ", uvals[obj_idx])
    # Here I redefine the box containing the polytope to be
    # the smallest posible, using the results of fva
    model["lb"] .= lvals
    model["ub"] .= uvals
    
    println("\nFVA pp model FBA")
    fva_pp_fbaout = fba_method(model)
    println("obj_val: ", fva_pp_fbaout.obj_val)
    println("flux sum: ", sum(fva_pp_fbaout.sol))
    println("error sum: ", sum(abs.(model["S"] * fva_pp_fbaout.sol - model["b"])))
    println()
    flush(stdout)
    return orig_fbaout, fva_pp_fbaout
end

fva_tests = Dict()
for model_file in model_files # HumanGEM takes a really long time
    fva_tests[model_file] = Dict()
    println("\n ------------- $model_file -------------")
    for (tname, fva_JuMP, fba_fun) in [("FVA JuMP-GLPK", fva_JuMP, fba_JuMP), 
                            ("FVA MathProgBase-Clp", fva_MathProgBase, fba_MathProgBase)]
        fva_tests[model_file][tname] = fva_pp_model_test(tname, fva_JuMP, fba_fun, model_file)
    end
end

for (model_file, results) in fva_tests
    @testset "$model_file" begin
        orig_obj_vals = [orig_fbaout.obj_val for (tname, (orig_fbaout, fva_pp_fbaout)) in results]
        # all original fba solutions must be the same
        @test all(isapprox.(orig_obj_vals |> first, orig_obj_vals))
        # all fba solutions, after fva preprocessing,  must be the same
        fva_pp_obj_vals = [fva_pp_fbaout.obj_val for (tname, (orig_fbaout, fva_pp_fbaout)) in results]
        @test all(isapprox.(fva_pp_obj_vals |> first, fva_pp_obj_vals))
        # all fba solutions must be the same
        all_obj_vals = [orig_obj_vals; fva_pp_obj_vals]
        @test all(isapprox.(all_obj_vals |> first, all_obj_vals))
    end
end

ps = []
for (model_file, results) in fva_tests
    p = Plots.plot(title = basename(model_file), xlabel = "FBA orig model", ylabel = "FBA fva preprocess model")
    for (tname, (orig_fbaout, fva_pp_fbaout)) in results
        Plots.scatter!(orig_fbaout.sol, fva_pp_fbaout.sol, label = tname, ms = 10)
    end
    push!(ps, p)
end

p = Plots.plot(ps..., size = (700, 300), layout = (1,length(ps)), titlefont = 10)

# ### Testing fva step by step


