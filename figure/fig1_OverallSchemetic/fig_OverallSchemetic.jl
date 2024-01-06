
using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using OrdinaryDiffEq
using Plots
using Statistics
using DataDrivenSparse
using ProgressLogging
using Symbolics
using Optimization
using OptimizationPolyalgorithms
using Combinatorics, Transducers
using OptimizationOptimJL
using ForwardDiff
using JLD2
using Setfield
using StatsBase
using BenchmarkTools
using SciMLSensitivity
using Flux
using Zygote
using DataFrames, CSV
using Hyperopt
import ColorSchemes.tab20
import ColorSchemes.grayC100
import ColorSchemes.seaborn_bright
import ColorSchemes.RdGy_11
import ColorSchemes.OrRd
using LaTeXStrings

cd(Base.source_path()*"/..")

include("../../ref_model/ref_model_hvvar.jl")
using .RefModel


#################### Part 1: generate reference data ####################

ref_model = ChemSys()
nspecs = 11

equations(convert(ODESystem, ref_model.rn))

ndays = 3
timelength = 60 * (ndays * 24) # minutes

saveat = 60.0 # minutes

nruns = 3000
ref_data_all = []
ref_params_all = []


for idx in 0:0
    JLD2.@load "../../dataset/ref_data_train.jld" ref_data_train ref_params_train times_train
    push!(ref_data_all, ref_data_train)
    push!(ref_params_all, ref_params_train)
    global times = times_train
end

ref_data = cat(ref_data_all...;dims=3)
ref_params = vcat(ref_params_all...)
ref_mean = mean(Array(ref_data), dims=(2,3))[:]


#begin
#for idx in 1:3
idx = 3
λ = [7.408656834939568e-6, 5.074705239490476e-6, 4.010572880855496e-5, 2.7471207892708143e-5][idx]
ε = [8.62723729246145e-5, 5.169241873891593e-6, 2.5049415421745025e-5, 1.0092715146305697e-6][idx]
β = [5.202404809619239, 3.2605210420841684 , 3.1843687374749496, 5.754509018036073][idx]
n_components = [1, 2, 3, 4][idx]

B = reshape((Array(ref_data) .- ref_mean) , nspecs, :)
B_ = copy(B)
B_[6,:] .=  B_[6,:].*β
B = B_
F = svd(B)

# Functions related to reference model.
#begin
    
function gen_emit(emit, hv_shift, initphase_2)
    emit_1(t) = emit
    emit_2(t) = emit*sin(t/720*pi - initphase_2) 
    emit_func(t) = 0.95*emit_1(t) + 0.05*emit_2(t)
    return emit_func
end

function emit_func(emit, hv_shift, initphase_2, t)

    emit_1 = emit
    emit_2 = emit*sin(t/720*pi - initphase_2) 
    emit = 0.95*emit_1  + 0.05*emit_2
    return emit
end

function gen_hv(hv_shift)
    hv_1(t) = max(sin((t+hv_shift)/720*pi - pi/2),0.0)
    hv_2(t) = max((sin((t+hv_shift)/720*pi - pi/2) + 1.0) / 2.0, 0.0)
    hv_func(t) = hv_1(t) * 0.9 + hv_2(t) * 0.1
    return hv_func
end

function hv_func(hv_shift, t)
    hv_1 = max(sin((t+hv_shift)/720*pi - pi/2),0.0)
    hv_2 = max((sin((t+hv_shift)/720*pi - pi/2) + 1.0) / 2.0, 0.0)
    hv_1 * 0.9 + hv_2 * 0.1
end

    rkppmmin(rk1,press,tempk) = rk1 * 60.0 *  7.34e+15 * press / tempk
    # Reaction rate constants
    rk2(press, tempk) = rkppmmin(6.0e-34 * ( tempk/300. )^(-2.3) * 7.34e+21 * press / tempk, press, tempk)
    rk3(press, tempk) = rkppmmin(2.e-12 * exp(-1400/tempk), press, tempk)
    rk6(press, tempk) = rkppmmin(1.2e-14 * tempk^1 * exp(+287/tempk), press, tempk)
    rk7(press, tempk) = rkppmmin(3.7e-12 * exp(+250/tempk), press, tempk)
    function rk8(press, tempk)
        rk0 = 2.0e-30 * ( tempk/300 )^(-3.0)
        rki = 2.5e-11 * ( tempk/300 )^0
        rm = 7.34e+21 * press / tempk
        rb = 0.6^(1.0 / ( 1.0 + (log10(rk0*rm/rki))^2.0 ))
        rk8 = rk0 * rm * rb / (1.0 + rk0*rm/rki)
        rk8 = rkppmmin(rk8,press,tempk)
    end
    rk10(press, tempk) = rkppmmin(3.3e-12 * exp(-200.0/tempk),press,tempk)
#end






function run_success(ode_prob, forcing_params, data_prob, all_params; kwargs...)
    prob2 = prob_func(ode_prob, forcing_params, all_params, data_prob.X[:,1])
    #display(prob2.p)
    estimate = solve(prob2, Rosenbrock23(), saveat=saveat)

end

function run_successes(ode_prob, ref_params, probs, all_params)
   nsuc = 0
   ind = []
   for i ∈ 1:nruns
        estimate = run_success(ode_prob, ref_params[i], probs.probs[i], all_params)
        if estimate.retcode != :Success
            nsuc+=1
            push!(ind,i)
        end
   end
   nsuc,ind
end


#run_successes(ode_prob, ref_params, probs, ode_prob.p)



function rmse(ode_prob, ref_params, probs, all_params, nruns)
    estimates = [] 
    truedata = []
    #rmse_sum = 0.0
    for i ∈ 1:nruns
        estimate = solve_rmse(ode_prob, ref_params[i], probs.probs[i], all_params)
        push!(truedata,Array(probs.probs[i].X))
        push!(estimates,Array(estimate)) #
    end
    truedata = permutedims(reshape(vcat(truedata...),(n_components,nruns,:)),(1,3,2)) 
    estimates = permutedims(reshape(vcat(estimates...),(n_components,nruns,:)),(1,3,2))#
    RMSETrain = rmsd(truedata, estimates)
end
#RMSETrain_ = rmse(ode_prob, ref_params, probs, ode_prob.p, nruns)

#push!(RMSETrain,RMSETrain_)




function run_rollout(ode_prob, forcing_params, data_prob, all_params; kwargs...)
    prob2 = prob_func(ode_prob, forcing_params, all_params, data_prob.X[:,1])
    #display(prob2.p)
    estimate = solve(prob2, Rosenbrock23(), saveat=saveat)
end
function run_rollouts(ode_prob, ref_params, probs, all_params)
   for i ∈ 1:nruns
    run_rollout(ode_prob, ref_params[i], probs.probs[i], all_params)
   end
end



function prob_func_finetuning(prob, forcing_params, all_params, u0)
    x0 = [u[j]=>u0[j] for j in 1:n_components]
    p = Zygote.Buffer(all_params)
    p[i_press] = forcing_params[1]
    p[i_tempk] = forcing_params[2]
    p[i_emission1_init] = forcing_params[3]
    p[i_emission2_init] = forcing_params[4]
    p[i_emission3_init] = forcing_params[5]
    p[i_emit1_initphase_2] = forcing_params[6]
    p[i_emit2_initphase_2] = forcing_params[7]
    p[i_emit3_initphase_2] = forcing_params[8]
    p[i_hv_shift] = forcing_params[9]
    pp = vcat(p[1:length(forcing_params)],all_params[(length(forcing_params)+1):end])
    #println("length(forcing_params) = $(length(forcing_params)) \n")
    #println("pp_ft = $(pp) \n")
    remake(prob, 
        u0 = u0,#ModelingToolkit.varmap_to_vars(x0, states(simple_sys)),
        p = pp,
    )
    
end


#RMSETest_FineTuned_ = rmse(ode_prob_test, ref_params_test, probs_test, p_FineTuned,nruns_test)

#push!(RMSETest_FineTuned,RMSETest_FineTuned_)

#################### Part 6: Decode to original coordinate system ####################

function plot_rollout_decoded(ode_prob, forcing_params, data_prob, ref_data, all_params; kwargs...)
    prob2 = prob_func(ode_prob, forcing_params, all_params, data_prob.X[:,1])
    estimate = solve(prob2, Rosenbrock23(), saveat=saveat)
    decoded = (F.U[:, 1:n_components] * Array(estimate)) .+ ref_mean
    #rmse = StatsBase.rmsd(decoded,ref_data)
    plot(estimate.t, ref_data', linecolor=[:black :red :blue],xlabel="Time (s)", ylabel="Conc. (ppm)" ; kwargs...)
    plot!(estimate.t, decoded', linestyle=:dash, linecolor=[:black :red :blue]; kwargs...)
    plot!(estimate.t, estimate[hv], label=hv, linecolor=:gray80; kwargs...)
end

function plot_rollouts_decoded(ode_prob, times, ref_params, probs, ref_data, all_params, species)
   p = []
   results = run_ensemble(ode_prob, ref_params, probs.probs, all_params)
   for i ∈ 1:16
        decoded = (F.U[:, 1:n_components] * Array(results[i]))
        decoded[6,:] = decoded[6,:]./β
        decoded = decoded .+ ref_mean
        decoded = copy(decoded[species,:])
        
        p1 = plot(times, ref_data[species, :, i]', 
            label=specs[:,species], 
            legend=(i==1 ? :best : :none), 
            linecolor=tab20[1:11]', 
            margin = 8Plots.mm,
            xlabel="Time (s)", ylabel="Conc. (ppm)" )
        plot!(results[i].t, decoded', label= :none, linestyle=:dash, linecolor=tab20[1:11]', margin = 8Plots.mm)
        #plot!(results[i].t, results[i][hv]./10, label=hv, linecolor=:gray80)
        push!(p, p1)
    end
   plot(p..., size=(1500, 1000))
end

# Decoded testing data

#plot_rollouts_decoded(ode_prob_test, times_test, ref_params_test, probs_test, ref_data_test, p_FineTuned)
function run_pca(ode_prob, ref_params, probs, all_params,nruns)
    estimates = [] 
    truedata = []
    #rmse_sum = 0.0
    for i ∈ 1:nruns
        estimate = solve_rmse(ode_prob, ref_params[i], probs.probs[i], all_params)
        push!(truedata,Array(probs.probs[i].X))
        push!(estimates,Array(estimate)) #
    end
    truedata = permutedims(reshape(vcat(truedata...),(n_components,nruns,:)),(1,3,2)) 
    estimates = permutedims(reshape(vcat(estimates...),(n_components,nruns,:)),(1,3,2))#
    RMSETrain = rmsd(truedata, estimates)
    return truedata, estimates
end
        
        



function rmse_decoded(ode_prob, ref_params, probs, all_params,nruns, specs)
    estimates = [] 
    truedata = []
    rmse_sum = 0.0
    for i ∈ 1:nruns
        estimate = solve_rmse(ode_prob, ref_params[i], probs.probs[i], all_params)
        
        truedata_ = (F.U[:, 1:n_components] * Array(probs.probs[i].X)) .+ ref_mean
        decoded = (F.U[:, 1:n_components] * Array(estimate))
        decoded[6,:] = decoded[6,:]./β
        decoded = decoded .+ ref_mean
    
        push!(truedata,Array(truedata_))#
            
        push!(estimates,Array(decoded)) #
    end
    truedata = permutedims(reshape(vcat(truedata...),(nspecs ,nruns,:)),(1,3,2))[specs,:,:]
    estimates = permutedims(reshape(vcat(estimates...),(nspecs ,nruns,:)),(1,3,2))[specs,:,:]
    
    truedata = reshape(truedata, (size(truedata)[1]*size(truedata)[2],))
    estimates = reshape(estimates, (size(estimates)[1]*size(estimates)[2],))
    
    #RMSE = rmsd(collect(StatsBase.trim(truedata, prop=0.3)), collect(StatsBase.trim(estimates, prop=0.3)))
    RMSE = Flux.mae(truedata, estimates)

end


function run_ensemble(prob, forcing_params, data_probs, all_params)   
    function setup(prob,i,repeat)
        u0 = data_probs[i].X[:,1]
        prob_func(prob, forcing_params[i], all_params, u0)
    end

    ensemble_prob = EnsembleProblem(prob, prob_func=setup)

    res = solve(ensemble_prob, Rosenbrock23(),EnsembleSerial(),trajectories=length(forcing_params), 
        saveat=saveat)
end








plot_case_medians = []
global rmsebycases = []
global ref6s = []
global pred6s = []
global median_inds = []
global var6s = ones(4,3)


    flag_plot = 1


    function create_prob(ref_data, ref_params, ref_mean, F, n_components, times, i)
        press, tempk, emission1_init, emission2_init, emission3_init, emit1_initphase_2, emit2_initphase_2, emit3_initphase_2, hv_shift = ref_params[i]
        hv_func = gen_hv(hv_shift)
        emit1func = gen_emit(emission1_init, hv_shift, emit1_initphase_2)
        emit2func = gen_emit(emission2_init, hv_shift, emit2_initphase_2)
        emit3func = gen_emit(emission3_init, hv_shift, emit3_initphase_2)
        s = (Array(ref_data[:, :, i]) .- ref_mean) #./ ref_std
        s[6,:,:] .=  s[6,:,:] .* β
        X = (F.U[:, 1:n_components]' * s)
        nc =  1.0e-4 # Constant for rate normalization for gradient descent.
        (X = X, t = times,
            U = (u, p, t) -> [emit1func(t), emit2func(t), emit3func(t), hv_func(t)],
            p=[press, tempk, emission1_init, emission2_init, emission3_init, 
            emit1_initphase_2, emit2_initphase_2, emit3_initphase_2,hv_shift]
        )
    end

    function solve_rmse(ode_prob, forcing_params, data_prob, all_params; kwargs...)
        prob2 = prob_func(ode_prob, forcing_params, all_params, data_prob.X[:,1])
        estimate = solve(prob2, Rosenbrock23(), saveat=saveat)
    end
    
    function prob_func(prob, forcing_params, all_params, u0)
        x0 = [u[j]=>u0[j] for j in 1:n_components]
        p = copy(all_params)
        p[i_press] = forcing_params[1]
        p[i_tempk] = forcing_params[2]
        p[i_emission1_init] = forcing_params[3]
        p[i_emission2_init] = forcing_params[4]
        p[i_emission3_init] = forcing_params[5]
        p[i_emit1_initphase_2] = forcing_params[6]
        p[i_emit2_initphase_2] = forcing_params[7]
        p[i_emit3_initphase_2] = forcing_params[8]
        p[i_hv_shift] = forcing_params[9]
    
        remake(prob, 
            u0 = ModelingToolkit.varmap_to_vars(x0, states(simple_sys)),
            p = p,
        )
    end
    
    function run_decoded(ode_prob, ref_params, probs, all_params,nruns)
        estimates = [] 
        #truedata = []
        #rmse_sum = 0.0
        for i ∈ 1:nruns
            estimate = solve_rmse(ode_prob, ref_params[i], probs.probs[i], all_params)
            
            #truedata_ = (F.U[:, 1:n_components] * Array(probs.probs[i].X)) .+ ref_mean
            decoded = (F.U[:, 1:n_components] * Array(estimate))
            decoded[6,:] = decoded[6,:]./β
            decoded = decoded .+ ref_mean
        
            #push!(truedata,Array(truedata_))#
                
            push!(estimates,Array(decoded)) #
        end
        #truedata = permutedims(reshape(vcat(truedata...),(nspecs ,nruns,:)),(1,3,2)) 
        estimates = permutedims(reshape(vcat(estimates...),(nspecs ,nruns,:)),(1,3,2))#
        #RMSE = rmsd(truedata, estimates)
    end
#for loop in 1:nloop
#################### Part 2: Use SINDy to find the model equations ####################
@parameters t
@variables u(t)[1:n_components]
@parameters press tempk hv_shift emit1_initphase_2 emit2_initphase_2 emit3_initphase_2 emission1_init emission2_init emission3_init
@parameters emission1 emission2 emission3 hv
u = collect(u)
#rk = collect(rk)
D = Differential(t)
# These are the candidate terms for the equations.
#u_div_combinations = reduce.(/, vcat([collect(combinations(u, i)) for i in 1:length(u)]...))
#u_mul_combinations = reduce.(*, vcat([collect(combinations(u, i)) for i in 1:length(u)]...))
#emission_mul_combinations = reduce.(/, vcat([collect(combinations(u, i)) for i in 1:length(u)]...))


h = Num[

    -0.4/(100*12) * 60; 
    -4/(100*15)* 60; 
    -0.5/(100*2) * 60; 
    -0.1/(100*2) * 60; 
    -0.03/2 * 60; 
    -0.1/100 * 60;
    0.001*emission1;
    0.1*emission2;
    0.001*emission3;
    polynomial_basis(u,3);
    polynomial_basis(u,2).*tempk*press;
    polynomial_basis(u,2).*hv;
]

JLD2.@load "../../sindy_model/model_system_params_3000_ncomp_$(idx).jld" system params F ref_mean


sys_eqn = (equations(system))
nexp = 3
if nexp%2==0
    expo = nexp+1
else
    expo = nexp +2
end

for i in 1:n_components
   sys_eqn = @set sys_eqn[i].rhs +=  - ε*(u[i])^expo
end

sys_eqn
params
@variables hv emission1 emission2 emission3# These were parameters above but now we need them to be variables.
nc =  1.0e-4 # Constant from above used for rate normalization for gradient descent.
eqs = [
    (sys_eqn);
    hv ~ hv_func(hv_shift, t);
    emission1 ~ emit_func(emission1_init, hv_shift, emit1_initphase_2, t)
    emission2 ~ emit_func(emission2_init, hv_shift, emit2_initphase_2, t)
    emission3 ~ emit_func(emission3_init, hv_shift, emit3_initphase_2, t)

]


@named sys = ODESystem(
    eqs,
    get_iv(system),
    states(system),
    [setdiff(parameters(system), [hv]);],
)
simple_sys = structural_simplify(sys)

ref_data
ps = [get_parameter_map(system); ]
#ps = get_parameter_map(system)
ode_prob = ODEProblem(simple_sys, zeros(n_components), (times[1], times[end]),ps)
#JLD2.jldsave("../main/model/mymodel_$(loop).jld";ref_params,ref_data,timesteps_keep,times,want_specs,time_,system,sys,ps,res,rss_val)
indexof(sym,syms) = findfirst(isequal(sym),syms)
i_press = indexof(press, parameters(sys))
i_tempk = indexof(tempk, parameters(sys))
i_emission1_init = indexof(emission1_init, parameters(sys))
i_emission2_init = indexof(emission2_init, parameters(sys))
i_emission3_init = indexof(emission3_init, parameters(sys))
i_emit1_initphase_2 = indexof(emit1_initphase_2, parameters(sys))
i_emit2_initphase_2 = indexof(emit2_initphase_2, parameters(sys))
i_emit3_initphase_2 = indexof(emit3_initphase_2, parameters(sys))
i_hv_shift = indexof(hv_shift, parameters(sys))

function plot_rollout(ode_prob, forcing_params, data_prob, all_params; kwargs...)
    prob2 = prob_func(ode_prob, forcing_params, all_params, data_prob.X[:,1])
    #display(prob2.p)
    estimate = solve(prob2, Rosenbrock23(),saveat=saveat)
    p = plot(data_prob.t, data_prob.X',
        label=reshape(["y$(i)" for i in 1:n_components],(1,n_components)), 
        linecolor=[:black :red :blue], 
        linewidth = 5,
        xlabel = "Time", ylabel = "Concentration", 
        xticks = false, yticks = false,
        xlabelfontsize = 25, ylabelfontsize = 25,
        left_margin = 4Plots.mm,
        legend = false, grid = false)
    return p
end
transpose(ref_data[:,:,1].-ref_mean)

conc_raw = plot(times, transpose(ref_data[:,:,181]), 
                grid=false, legend=false, 
                xticks=false, yticks=false,
                xlabel = "Time", ylabel="Concentration",
                xlabelfontsize = 25, ylabelfontsize = 25,
                #palette = :seaborn_bright,
                linecolor=[:orange seaborn_bright[2] :red seaborn_bright[3] seaborn_bright[4] :blue :black seaborn_bright[5] seaborn_bright[6] seaborn_bright[7] seaborn_bright[8]], 
                linewidth = 5,
                left_margin = 4Plots.mm
                )

probnames = Tuple(Symbol.(["prob$i" for i in 1:nruns]));
probdata = Tuple([create_prob(ref_data, ref_params, ref_mean, F, n_components, times, i) for i in 1:nruns]);
probtuple = NamedTuple{probnames}(probdata);
global probs = DataDrivenDiffEq.ContinuousDataset(probtuple);

conc_latent = plot_rollout(ode_prob, ref_params[181], probs.probs[181], ode_prob.p)


savefig(conc_raw, "ref_conc_raw.svg")
savefig(conc_latent, "ref_conc_latent.svg")

want_specs = setdiff(1:13, [ref_model.specmap["O2"], ref_model.specmap["H2O"]])
specs = reshape((states(ref_model.rn)[want_specs]),(1,11))
specs_ = specs
specs_ = [string(i)[1:(end-3)] for i in specs ]
#n_components = 11
a = F.U[:, 1:nspecs]
gr()
heatplot = heatmap(a,xticks=(1:nspecs),yticks=(1:11,specs_), 
                   showaxis = false,
                   c = :RdBu_9, legend = :none,
                   size = (400,400),
                   left_margin = -11.3Plots.mm,
                   right_margin = -0.5Plots.mm,
                   top_margin = -1.5Plots.mm,
                   bottom_margin = -6Plots.mm)
fontsize = 7
nrow, ncol = size(a)
ann = [(i,j, text(round(a[i,j], digits=2), fontsize, :white, :center))
            for i in 1:nrow for j in 1:ncol]
#annotate!(ann, linecolor=:white)
annotate!([(j, i, text(round(a[i,j],digits=3), fontsize, :white)) for i in 1:nrow for j in 1:ncol])

heatplot

using Latexify
latexify(sys_eqn) |> print
sys_eqn
render(latexify(sys_eqn))
println(h)
params


sparse_csv = CSV.read("sparse.csv", DataFrame, header=true)
sparse_coef = Array(sparse_csv)[:,2:end]
sparse_coef_filtered = similar(sparse_coef)
for i in 1:size(sparse_coef)[1]
    for j in 1:size(sparse_coef)[2]
        if sparse_coef[i,j] != 0.0
            sparse_coef_filtered[i,j] = 1
        else
            sparse_coef_filtered[i,j] = 0
        end
    end
end

sparse_coef_filtered
sparse_coef_filtered[:,1:1]
sparse_coef_heatplot_1 = heatmap(transpose(sparse_coef_filtered[:,1:1]),
                   xticks=(1:1),
                   #cscale = log10,
                   clim=(-1,1),
                   #yticks=(1:11,specs_), 
                   showaxis = false,
                   c = cgrad([:white, :red], 2, categorical = true), legend = :none,
                   size = (400,40),
                   left_margin = -10.2Plots.mm,
                   right_margin = -1.8Plots.mm,
                   top_margin = -1.8Plots.mm,
                   bottom_margin = -3.2Plots.mm
                   )

sparse_coef_heatplot_2 = heatmap(transpose(sparse_coef_filtered[:,2:2]),
                   xticks=(1:1),
                   #cscale = log10,
                   clim=(-1,1),
                   #yticks=(1:11,specs_), 
                   showaxis = false,
                   c = cgrad([:white, :blue], 2, categorical = true), legend = :none,
                   size = (400,40),
                   left_margin = -10.2Plots.mm,
                   right_margin = -1.8Plots.mm,
                   top_margin = -1.8Plots.mm,
                   bottom_margin = -3.2Plots.mm
                   )

sparse_coef_heatplot_3 = heatmap(transpose(sparse_coef_filtered[:,3:3]),
                   xticks=(1:1),
                   #cscale = log10,
                   clim=(-1,1),
                   #yticks=(1:11,specs_), 
                   showaxis = false,
                   c = cgrad([:white, :black], 2, categorical = true), legend = :none,
                   size = (400,40),
                   left_margin = -10.2Plots.mm,
                   right_margin = -1.8Plots.mm,
                   top_margin = -1.8Plots.mm,
                   bottom_margin = -3.2Plots.mm
                   )

savefig(sparse_coef_heatplot_1, "sparse_coeff_1.svg")
savefig(sparse_coef_heatplot_2, "sparse_coeff_2.svg")
savefig(sparse_coef_heatplot_3, "sparse_coeff_3.svg")

sparse_coef_num_ref = similar(sparse_coef[:,1:1])

for i in 1:size(sparse_coef)[1]
    if i ∈ 1:3 || i ∈ 39:58
        sparse_coef_num_ref[i,1] = 1
    else
        sparse_coef_num_ref[i,1] = 0
    end
end

sparse_coef_num_ref

sparse_coef_num_ref_plot = heatmap(transpose(sparse_coef_num_ref[:,1:1]),
                   xticks=(1:1),
                   #cscale = log10,
                   clim=(-1,1),
                   #yticks=(1:11,specs_), 
                   showaxis = false,
                   c = cgrad([:white, :green], 2, categorical = true), legend = :none,
                   size = (400,40),
                   left_margin = -10.2Plots.mm,
                   right_margin = -1.8Plots.mm,
                   top_margin = -1.8Plots.mm,
                   bottom_margin = -3.2Plots.mm
                   )
plot_sparse_coeff = plot(sparse_coef_heatplot_1, sparse_coef_heatplot_2, sparse_coef_heatplot_3, layout=(3,1),size=(400,400))
savefig(plot_sparse_coeff, "sparse_coeff.svg")


function plot_rollout(ode_prob, forcing_params, data_prob, all_params; kwargs...)
    prob2 = prob_func(ode_prob, forcing_params, all_params, data_prob.X[:,1])
    #display(prob2.p)
    estimate = solve(prob2, Rosenbrock23(),saveat=saveat)
    p = plot( data_prob.t, data_prob.X',
    label=reshape(["y$(i)" for i in 1:n_components],(1,n_components)), 
    linecolor=[:black :red :blue], 
    linewidth = 5, 
    xlabel = "Time", ylabel = "Concentration", 
    xticks = false, yticks = false,
    xlabelfontsize = 25, ylabelfontsize = 25,
    left_margin = 4Plots.mm,
    legend = false, grid = false)

    p = plot( estimate,
        label=reshape(["y$(i)" for i in 1:n_components],(1,n_components)), 
        linecolor=[:black :red :blue], 
        linewidth = 5, linestyle = :dash,
        xlabel = "Time", ylabel = "Concentration", 
        xticks = false, yticks = false,
        xlabelfontsize = 25, ylabelfontsize = 25,
        left_margin = 4Plots.mm,
        legend = false, grid = false)
    return p
end


pred_conc_latent = plot_rollout(ode_prob, ref_params[181], probs.probs[181], ode_prob.p)
savefig(pred_conc_latent, "pred_conc_latent.svg")


nruns_test = 375
ndays_test = 10

    JLD2.@load "../../dataset/ref_data_test.jld" ref_data_test ref_params_test times_test
    #ref_data_test =  ref_data
    #ref_params_test =  ref_params
    #times_test = times
    probnames_test = Tuple(Symbol.(["prob$i" for i in 1:nruns_test]));
    probdata_test = Tuple([create_prob(ref_data_test, ref_params_test, ref_mean, F, n_components, times_test, i) for i in 1:nruns_test]);
    probtuple_test = NamedTuple{probnames_test}(probdata_test);
    probs_test = DataDrivenDiffEq.ContinuousDataset(probtuple_test);


ode_prob_test = ODEProblem(simple_sys, zeros(n_components), (times_test[1], times_test[end]), ps)

function plot_rollout_decoded(ode_prob, forcing_params, data_prob, ref_data, all_params; kwargs...)
    prob2 = prob_func(ode_prob, forcing_params, all_params, data_prob.X[:,1])
    estimate = solve(prob2, Rosenbrock23(), saveat=saveat)
    decoded = (F.U[:, 1:n_components] * Array(estimate)) .+ ref_mean
    #rmse = StatsBase.rmsd(decoded,ref_data)
    plot(estimate.t, ref_data', linecolor=[:black :red :blue],xlabel="Time (s)", ylabel="Conc. (ppm)" ; kwargs...)
    plot!(estimate.t, decoded', linestyle=:dash, linecolor=[:black :red :blue]; kwargs...)
    plot!(estimate.t, estimate[hv], label=hv, linecolor=:gray80; kwargs...)
end

function plot_rollouts_decoded(ode_prob, times, ref_params, probs, ref_data, all_params, species)
   p = []
   results = run_ensemble(ode_prob, ref_params, probs.probs, all_params)
   for i ∈ 181:181
        decoded = (F.U[:, 1:n_components] * Array(results[i]))
        decoded[6,:] = decoded[6,:]./β
        decoded = decoded .+ ref_mean
        decoded = copy(decoded[species,:])
        
        p1 = plot(results[i].t, decoded',
        grid=false, legend=false, 
        xticks=false, yticks=false,
        xlabel = "Time", ylabel="Concentration",
        xlabelfontsize = 25, ylabelfontsize = 25,
        linestyle = :dash,
        #palette = :seaborn_bright,
        linecolor=[:orange seaborn_bright[2] :red seaborn_bright[3] seaborn_bright[4] :blue :black seaborn_bright[5] seaborn_bright[6] seaborn_bright[7] seaborn_bright[8]], 
        linewidth = 5,
        left_margin = 4Plots.mm )
        #plot!(results[i].t, decoded', label= :none, linestyle=:dash, linecolor=tab20[1:11]', margin = 8Plots.mm)
        #plot!(results[i].t, results[i][hv]./10, label=hv, linecolor=:gray80)
        push!(p, p1)
    end
   plot(p[1])
end

pred_conc_raw = plot_rollouts_decoded(ode_prob, times, ref_params, probs, ref_data, ode_prob.p,1:11)
savefig(pred_conc_raw, "pred_conc_raw.svg")

function plot_rollouts_decoded(ode_prob, times, ref_params, probs, ref_data, all_params, species, case)
    p = []
    results = run_ensemble(ode_prob, ref_params, probs.probs, all_params)
    for i ∈ case:case
         decoded = (F.U[:, 1:n_components] * Array(results[i]))
         decoded[6,:] = decoded[6,:]./β
         decoded = decoded .+ ref_mean
         decoded = copy(decoded[species,:])
         p1 = plot(times, ref_data[species, :, i]',
         grid=false, legend=false, 
         xticks=false, yticks=false,
         xlabel = "Time", ylabel="Concentration",
         xlabelfontsize = 25, ylabelfontsize = 25,
         #linestyle = :dash,
         #palette = :seaborn_bright,
         linecolor=[:black], 
         linewidth = 3,
         left_margin = 13Plots.mm,
         bottom_margin = 8Plots.mm )
        
        plot!(results[i].t, decoded',
         grid=false, legend=false, 
         xticks=false, yticks=false,
         xlabel = "Time", ylabel="Concentration",
         xlabelfontsize = 25, ylabelfontsize = 25,
         linestyle = :dash,
         #palette = :seaborn_bright,
         linecolor=[:red ], 
         linewidth = 5,
         left_margin = 13Plots.mm,
         bottom_margin = 8Plots.mm )

         
         #plot!(results[i].t, results[i][hv]./10, label=hv, linecolor=:gray80)
         push!(p, p1)
     end
     plot(p[1],size=(1200,400))
end

plot_rollouts_decoded(ode_prob_test, times_test, ref_params_test, probs_test, ref_data_test, ode_prob_test.p,6:6, 42)
plot_rollouts_decoded(ode_prob_test, times_test, ref_params_test, probs_test, ref_data_test, ode_prob_test.p,6:6, 181)
plot_rollouts_decoded(ode_prob_test, times_test, ref_params_test, probs_test, ref_data_test, ode_prob_test.p,6:6, 349)
