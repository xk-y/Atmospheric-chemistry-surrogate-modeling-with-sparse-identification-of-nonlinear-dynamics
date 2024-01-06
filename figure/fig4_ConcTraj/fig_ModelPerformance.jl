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
using Statistics
import ColorSchemes.tab20
import ColorSchemes.grayC100

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
idx = 0

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




n_pr = 6
idx = 3
n_components = [1, 2, 3, 4][idx]
λ=[7.408656834939568e-6, 5.074705239490476e-6, 4.010572880855496e-5, 2.7471207892708143e-5][idx]
ε=[8.62723729246145e-5, 5.169241873891593e-6, 2.5049415421845025e-5, 1.0092715146305697e-6][idx]
β=[5.202404809619239, 3.2605210420841684 , 3.1843687374749496, 5.754509018036073][idx]

#λ=3.40041193270371e-5
#ε=1.8979216428391035e-6
#β=3.248248248248248

ref_mean = mean(Array(ref_data), dims=(2,3))[:]
B = reshape((Array(ref_data) .- ref_mean) , nspecs, :)
B_ = copy(B)
B_[n_pr,:] .=  B_[n_pr,:].*β
B = B_
F = svd(B)

function create_prob(ref_data, ref_params, times, i)
    press, tempk, emission1_init, emission2_init, emission3_init, emit1_initphase_2, emit2_initphase_2, emit3_initphase_2, hv_shift = ref_params[i]
    hv_func = gen_hv(hv_shift)
    emit1func = gen_emit(emission1_init, hv_shift, emit1_initphase_2)
    emit2func = gen_emit(emission2_init, hv_shift, emit2_initphase_2)
    emit3func = gen_emit(emission3_init, hv_shift, emit3_initphase_2)
    s = (Array(ref_data[:, :, i]) .- ref_mean) 
    s[n_pr,:,:] .=  s[n_pr,:,:] .* β
    X = (F.U[:, 1:n_components]' * s)
    nc =  1.0e-4 # Constant for rate normalization for gradient descent.
    (X = X, t = times,
        U = (u, p, t) -> [emit1func(t), emit2func(t), emit3func(t), hv_func(t)],
        p=[press, tempk, emission1_init, emission2_init, emission3_init, 
        emit1_initphase_2, emit2_initphase_2, emit3_initphase_2,hv_shift]
    )
end

#probnames = Tuple(Symbol.(["prob$i" for i in 1:nruns]));
#probdata = Tuple([create_prob(ref_data, ref_params, times, i) for i in 1:nruns]);
#probtuple = NamedTuple{probnames}(probdata);
#probs = DataDrivenDiffEq.ContinuousDataset(probtuple);

want_specs = setdiff(1:13, [ref_model.specmap["O2"], ref_model.specmap["H2O"]])
specs = reshape((states(ref_model.rn)[want_specs]),(1,11))
specs = [string(i)[1:(end-3)] for i in specs ]


function solve_rmse(ode_prob, forcing_params, data_prob, all_params; kwargs...)
    prob2 = prob_func(ode_prob, forcing_params, all_params, data_prob.X[:,1])
    estimate = solve(prob2, Tsit5(), saveat=saveat)
end

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


function plot_rollout(ode_prob, forcing_params, data_prob, all_params; kwargs...)
    prob2 = prob_func(ode_prob, forcing_params, all_params, data_prob.X[:,1])
    estimate = solve(prob2, Tsit5(),saveat=saveat)
    plot(data_prob.t, data_prob.X',
        label=reshape(["y$(i)" for i in 1:n_components],(1,n_components)), 
        linecolor=[:black :red :blue :orange], xlabel = "Time (s)", ylabel = "Conc. (ppm)", margin = 8Plots.mm
        ; kwargs...)
    plot!(estimate, linestyle=:dash, label=reshape(["y$(i)_pred" for i in 1:n_components],(1,n_components)),
        linecolor=[:black :red :blue :orange], xlabel = "Time (s)", ylabel = "Conc. (ppm)", margin = 8Plots.mm
        ; kwargs...)
    plot!(estimate.t, estimate[hv]./10, label=hv, xlabel = "Time (s)", ylabel = "Conc. (ppm)",
        linecolor=:gray80, margin = 8Plots.mm
        ; kwargs...)
end

function plot_rollouts(ode_prob, ref_params, probs, all_params)
   p = [] 
   for i ∈ 1:16
        push!(p, plot_rollout(ode_prob, ref_params[i], probs.probs[i], all_params;
            legend=(i==1 ? :best : :none),legendfontsize=8))
   end
   plot(p...,size=(1500, 1000))
end

function run_rollout(ode_prob, forcing_params, data_prob, all_params; kwargs...)
    prob2 = prob_func(ode_prob, forcing_params, all_params, data_prob.X[:,1])
    #display(prob2.p)
    estimate = solve(prob2, Tsit5(), saveat=saveat)
end
function run_rollouts(ode_prob, ref_params, probs, all_params)
   for i ∈ 1:nruns
    run_rollout(ode_prob, ref_params[i], probs.probs[i], all_params)
   end
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

function plot_rollout_decoded(ode_prob, forcing_params, data_prob, ref_data, all_params; kwargs...)
    prob2 = prob_func(ode_prob, forcing_params, all_params, data_prob.X[:,1])
    estimate = solve(prob2, Tsit5(), saveat=saveat)
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
        decoded[n_pr,:] = decoded[n_pr,:]./β
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
        
        
function run_decoded(ode_prob, ref_params, probs, all_params,nruns)
    estimates = [] 
    #truedata = []
    #rmse_sum = 0.0
    for i ∈ 1:nruns
        estimate = solve_rmse(ode_prob, ref_params[i], probs.probs[i], all_params)
        
        #truedata_ = (F.U[:, 1:n_components] * Array(probs.probs[i].X)) .+ ref_mean
        decoded = (F.U[:, 1:n_components] * Array(estimate))
        decoded[n_pr,:] = decoded[n_pr,:]./β
        decoded = decoded .+ ref_mean
    
        #push!(truedata,Array(truedata_))#
            
        push!(estimates,Array(decoded)) #
    end
    #truedata = permutedims(reshape(vcat(truedata...),(nspecs ,nruns,:)),(1,3,2)) 
    estimates = permutedims(reshape(vcat(estimates...),(nspecs ,nruns,:)),(1,3,2))#
    #RMSE = rmsd(truedata, estimates)
end


function rmse_decoded(ode_prob, ref_params, probs, all_params, ref_data, nruns, specs)
    estimates = [] 
    truedata = []
    rmse_sum = 0.0
    for i ∈ 1:nruns
        estimate = solve_rmse(ode_prob, ref_params[i], probs.probs[i], all_params)
        
        truedata_ = (F.U[:, 1:n_components] * Array(probs.probs[i].X)) .+ ref_mean
        decoded = (F.U[:, 1:n_components] * Array(estimate))
        decoded[n_pr,:] = decoded[n_pr,:]./β
        decoded = decoded .+ ref_mean
    
        push!(truedata,Array(truedata_))#
            
        push!(estimates,Array(decoded)) #
    end
    truedata = ref_data[specs,:,:]
    estimates = permutedims(reshape(vcat(estimates...),(nspecs ,nruns,:)),(1,3,2))[specs,:,:]
    
    truedata = reshape(truedata, (size(truedata)[1]*size(truedata)[2],))
    estimates = reshape(estimates, (size(estimates)[1]*size(estimates)[2],))
    ref_data
    RMSE = rmsd(truedata, estimates)
    #truedata#, estimates
end


function run_ensemble(prob, forcing_params, data_probs, all_params)   
    function setup(prob,i,repeat)
        u0 = data_probs[i].X[:,1]
        prob_func(prob, forcing_params[i], all_params, u0)
    end

    ensemble_prob = EnsembleProblem(prob, prob_func=setup)

    res = solve(ensemble_prob, Tsit5(),EnsembleSerial(),trajectories=length(forcing_params), 
        saveat=saveat)
end


@parameters t
@variables u(t)[1:n_components]
@parameters press tempk hv_shift emit1_initphase_2 emit2_initphase_2 emit3_initphase_2 emission1_init emission2_init emission3_init
@parameters emission1 emission2 emission3 hv
u = collect(u)

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

basis = Basis(h, u, parameters = [press, tempk,
                emission1_init, emission2_init, emission3_init, emit1_initphase_2, emit2_initphase_2, emit3_initphase_2, hv_shift], 
                controls = [emission1, emission2, emission3, hv])

#opt = STLSQ(λ)
#tmp_res = solve(probs, basis, opt)

#global rss_val = rss(tmp_res)
#global res = tmp_res

#system = get_basis(res)
#params = get_parameter_map(system)
JLD2.@load "../../sindy_model/model_system_params_3000_ncomp_$(n_components).jld" system params F ref_mean


sys_eqn = (equations(system))
nexp = 3
if nexp%2==0
    expo = nexp+1
else
    expo = nexp +2
end

for i in 1:n_components
   global sys_eqn = @set sys_eqn[i].rhs +=  - ε*(u[i])^expo
end


@variables hv emission1 emission2 emission3
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

ps = [get_parameter_map(system); ]
ode_prob = ODEProblem(simple_sys, zeros(n_components), (times[1], times[end]),ps)

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



#ref_data_compressed = []
#for i in 1:nruns
#    push!(ref_data_compressed, probs.probs[i].X)
#end
#ref_data_compressed = permutedims(reshape(vcat(ref_data_compressed...),(n_components,nruns,:)),(1,3,2)) 


nruns_test = 375
ndays_test = 10

JLD2.@load "../../dataset/ref_data_test.jld" ref_data_test ref_params_test times_test
probnames_test = Tuple(Symbol.(["prob$i" for i in 1:nruns_test]));
probdata_test = Tuple([create_prob(ref_data_test, ref_params_test, times_test, i) for i in 1:nruns_test]);
probtuple_test = NamedTuple{probnames_test}(probdata_test);
probs_test = DataDrivenDiffEq.ContinuousDataset(probtuple_test);

ode_prob_test = ODEProblem(simple_sys, zeros(n_components), (times_test[1], times_test[end]), ps)

sqrt(sum((ref_data_test[6,:,:]).^2)/(nruns_test*216))
#test_prediction = run_decoded(ode_prob_test, ref_params_test, probs_test, ode_prob_test.p, nruns_test)
#ref_data_test[6,:,1]
#test_prediction[6,:,1]
#specs
#plot(ref_data_test[6,:,11])
#plot!(test_prediction[6,:,11])
#JLD2.jldsave("testing_perform.jld"; ref_data_test, test_prediction, specs)


pwd()
for ispec in 1:11
#num_uns,ind_uns = run_successes(ode_prob_test, ref_params_test, probs_test, ode_prob_test.p, nruns_test)
percent = 95
predd = run_decoded(ode_prob_test, ref_params_test, probs_test, ode_prob_test.p, nruns_test)
ref_data_test
pred6 = run_decoded(ode_prob_test, ref_params_test, probs_test, ode_prob_test.p, nruns_test)[ispec,:,:]
rmsd(predd,ref_data_test)

0.0549
ref6 = ref_data_test[ispec,:,:]
err6 = (pred6 - ref6)#./ref6
mean6 = mean(ref6;dims=2)
upper6, lower6 = [], []
for i in 1:length(times_test)
    push!(upper6, percentile(err6[i,:],percent+(100-percent)/2))
    push!(lower6, percentile(err6[i,:],(100-percent)/2))
end

pred6_90 = run_decoded(ode_prob_test, ref_params_test, probs_test, ode_prob_test.p, nruns_test)[ispec,:,:]
ref6_90 = ref_data_test[ispec,:,:]
err6_90 = (pred6_90 - ref6_90)#./ref6_90
mean6_90 = mean(ref6_90;dims=2)
upper6_90, lower6_90 = [], []
for i in 1:length(times_test)
    push!(upper6_90, percentile(err6_90[i,:],90+(100-90)/2))
    push!(lower6_90, percentile(err6_90[i,:],(100-90)/2))
end
        
pred6_100 = run_decoded(ode_prob_test, ref_params_test, probs_test, ode_prob_test.p, nruns_test)[ispec,:,:]
ref6_100 = ref_data_test[ispec,:,:]
err6_100 = (pred6_100 - ref6_100)#./ref6_100
mean6_100 = mean(ref6_100;dims=2)
upper6_100, lower6_100 = [], []
for i in 1:length(times_test)
    push!(upper6_100, percentile(err6_100[i,:],100+(100-100)/2))
    push!(lower6_100, percentile(err6_100[i,:],(100-100)/2))
end

pred6_80 = run_decoded(ode_prob_test, ref_params_test, probs_test, ode_prob_test.p, nruns_test)[ispec,:,:]
ref6_80 = ref_data_test[ispec,:,:]
err6_80 = (pred6_80 - ref6_80)#./ref6_80
mean6_80 = mean(ref6_80;dims=2)
upper6_80, lower6_80 = [], []
for i in 1:length(times_test)
    push!(upper6_80, percentile(err6_80[i,:],80+(100-80)/2))
    push!(lower6_80, percentile(err6_80[i,:],(100-80)/2))
end

import ColorSchemes.grayC100


rmsebycase = []

for i in 1:size(ref6,2)
    push!(rmsebycase, rmsd(ref6[:,i],pred6[:,i]))
end
rmsebycase
function plot_extreme_cases(scenario, err)

    function plot_case(ref, pred, time, ind, err_, scenario)
        p = [] 
        no = 1
        p_title = plot(title = "$(scenario)",titlefontsize = 20, grid = false, showaxis = false, bottom_margin = -160Plots.px)
        push!(p, p_title)
        for i in ind
            
            ptemp = plot(time./1440, ref[:,i];
                         legend=((no==1 && scenario=="Worst") ? :topright : :none), 
                         labels="Reference",
                         #title = "rmse = $(round(err_[i]; digits = 3))",
                         
                         xtickfontsize = 14, ytickfontsize = 14, 
                         xlabelfontsize = 18, ylabelfontsize = 18,
                         legendfontsize = 20,
                         xlim=(1,10),
                         #xlabel = (scenario == "Median" && no == 3 ? "Time (day)" : ""),
                         ylabel = (scenario == "Best" && no == 2 ? "$(specs[1,ispec]) (ppm)" : ""),
                         #ylim=((no==1 && scenario=="Worst") ? (0.0,0.335) : :best),
                         left_margin = (scenario == "Best" ? 8Plots.mm : 4Plots.mm ),
                         right_margin = (scenario == "Worst" ? 3Plots.mm : 1Plots.mm ),
                         bottom_margin = (no == 3 ? 7Plots.mm : 0Plots.mm ),
                         top_margin = (no == 1 ? 7Plots.mm : 3Plots.mm ),
                         color=:black,
                         formatter = identity)
            plot!(time./1440, pred[:,i]; 
                  labels="SINDy",
                  linestyle=:dash, color=:red)
            push!(p, ptemp)
            no+=1
        end
        plot(p...,size=(1200, 500),layout=(4,1))
    end

    if scenario == "Best"
        minicases = []
        local temp = err
        local inds = Array(1:length(err))
        for i in 1:3
                minicase = findmin(temp)[2]
                push!(minicases,inds[minicase])
                temp = temp[1:end .!= minicase]
                inds = inds[1:end .!= minicase]
        end
        println(minicases)
        plot_idx = setdiff(Array(1:length(err)), inds)
        return plotmin = plot_case(ref6, pred6, times_test, minicases, err, scenario)
    end

    if scenario == "Worst"
        maxicases = []
        local temp = err
        local inds = Array(1:length(err))
        for i in 1:3
                maxicase = findmax(temp)[2]
                push!(maxicases,inds[maxicase])
                temp = temp[1:end .!= maxicase]
                inds = inds[1:end .!= maxicase]
        end
        println(maxicases)
        plot_idx = setdiff(Array(1:length(err)), inds)
        return plotmax = plot_case(ref6, pred6, times_test, maxicases, err, scenario)
    end

    if scenario == "Median"
        rmsebycase_index = hcat(err,1:length(err))
        rmsebycase_index_order = hcat(rmsebycase_index,sortperm(rmsebycase))
        sorted_rmsebycase_index_order = sortslices(rmsebycase_index_order,dims=1,by=x->x[1],rev=false)
        median_ind = [sorted_rmsebycase_index_order[Int(floor(length(rmsebycase)/2))-1,2],
        sorted_rmsebycase_index_order[Int(floor(length(rmsebycase)/2)),2],
        sorted_rmsebycase_index_order[Int(floor(length(rmsebycase)/2))+1,2]]
        println(median_ind)
        return plotmax = plot_case(ref6, pred6, times_test, median_ind, err, scenario)
    end



end

plot_error =  plot(times_test./1440,mean(err6_100;dims=2); 
                   ribbon = (-lower6_100, upper6_100 ),
                   color = grayC100[20],
                   label="100%", 
                   xlabel="Time (day)",
                   ylabel = "$(specs[1,ispec]) (ppm)",
                   titlefontsize = 16,
                   xtickfontsize = 16, ytickfontsize = 16, 
                   xlabelfontsize = 18, ylabelfontsize = 18,
                   legendfontsize = 20,
                   linewidth=0,
                   #yticks=[-0.3, -0.2,-0.1,0,0.1,0.2,0.3],
                   left_margin = 8Plots.mm,
                   bottom_margin = 12Plots.mm,
                   formatter = identity,
                   size=(1200, 300)
                   )
#plot!(times_test./1440,mean(err6;dims=2); linewidth=0, ribbon = ( -lower6 , upper6), color = grayC100[40], label="95%")
plot!(times_test./1440,mean(err6_90;dims=2);linewidth=0, ribbon = (-lower6_90, upper6_90 ), color = grayC100[65], label="90%")
plot!(times_test./1440,mean(err6_80;dims=2); linewidth=0, ribbon = (-lower6_80, upper6_80 ), color = grayC100[100], label="80%")






plot_case_best = plot_extreme_cases("Best",(rmsebycase))
rmsebycase[42],rmsebycase[100],rmsebycase[146]

rmsebycase[100]/mean(ref_data_test)^0.5

plot_case_median = plot_extreme_cases("Median",(rmsebycase))
rmsebycase[181],rmsebycase[167],rmsebycase[372]
plot_case_worst = plot_extreme_cases("Worst",(rmsebycase))
rmsebycase[349],rmsebycase[327],rmsebycase[47]
plot_case_bmw = plot(plot_case_best, plot_case_median, plot_case_worst, layout=(1,3))

#p_title = plot(xlabel="Time (day)", 
#               bottom_margin = 10Plots.mm, 
#               left_margin = -4Plots.mm,
#               xlabelfontsize = 18, grid = false, showaxis = false)

#plot_case_bmw_err = plot(plot_case_bmw,p_title, plot_error,layout=grid(1,3, widths=(0.5,0.001,0.5)))


savefig(plot_case_bmw, "raw/fig5_nspec_$(specs[1,ispec])_ncomp_$(n_components)_case.svg")
savefig(plot_error, "raw/fig5_nspec_$(specs[1,ispec])_ncomp_$(n_components)_err.svg")
end

#plot(plot_case_bmw,plot_error,layout=(2,1),size = (1200,600))
#function rmse_decoded(ode_prob, ref_params, probs, all_params, ref_data, ref_mean, nruns, specs)
#    estimates = [] 
#    truedata = []
#    rmse_sum = 0.0
#    for i ∈ 1:nruns
#        estimate = solve_rmse(ode_prob, ref_params[i], probs.probs[i], all_params)
#        decoded = (F.U[:, 1:n_components] * Array(estimate))
#        decoded[6,:] = decoded[6,:]./β
#        decoded = decoded .+ ref_mean
#            
#        push!(estimates,Array(decoded)) #
#    end
#    truedata = ref_data[specs,:,:]
#    estimates = permutedims(reshape(vcat(estimates...),(nspecs ,nruns,:)),(1,3,2))[specs,:,:]
#    println(size(truedata),size(estimates))
#    truedata = reshape(truedata, (size(truedata)[1]*size(truedata)[2]*size(truedata)[3],))
#    estimates = reshape(estimates, (size(estimates)[1]*size(estimates)[2]*size(estimates)[3],))
#    RMSE = rmsd(truedata, estimates)
#end

#RMSETestDecoded = rmse_decoded(ode_prob_test, ref_params_test, probs_test, ode_prob_test.p, ref_data_test, ref_mean, nruns_test, 1:11)
1