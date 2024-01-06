using DataDrivenDiffEq
using LinearAlgebra
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


    
    function gen_emit(emit, hv_shift, initphase_2)
        emit_1(t) = emit
        emit_2(t) = emit*sin(t/720*pi - initphase_2) 
        emit_func(t) = 0.95*emit_1(t) + 0.05*emit_2(t)
        return emit_func
    end
    
    function emit_func(emit, hv_shift, initphase_2, t)
        #random.seed!(42)
        emit_1 = emit#max(emit*sin((t+hv_shift-720.0)/1440*2*pi - pi/2),0.0) # initphase_2 = 1440
        emit_2 = emit*sin(t/720*pi - initphase_2) # initphase_2 = 1440*7, initphase_2∈[0,2*pi]
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


RMSEs = []

for idx in 1:4

    n_components = [1, 2, 3, 4][idx]
    λ=[7.408656834939568e-6, 5.074705239490476e-6, 4.010572880855496e-5, 2.7471207892708143e-5][idx]
    ε=[8.62723729246145e-5, 5.169241873891593e-6, 2.5049415421745025e-5, 1.0092715146305697e-6][idx]
    β=[5.202404809619239, 3.2605210420841684 , 3.1843687374749496, 5.754509018036073][idx]
ref_mean = mean(Array(ref_data), dims=(2,3))[:]
B = reshape((Array(ref_data) .- ref_mean) , nspecs, :)
B_ = copy(B)
B_[6,:] .=  B_[6,:].*β
B = B_
F = svd(B)

function create_prob(ref_data, ref_params, times, i)
    press, tempk, emission1_init, emission2_init, emission3_init, emit1_initphase_2, emit2_initphase_2, emit3_initphase_2, hv_shift = ref_params[i]
    hv_func = gen_hv(hv_shift)
    emit1func = gen_emit(emission1_init, hv_shift, emit1_initphase_2)
    emit2func = gen_emit(emission2_init, hv_shift, emit2_initphase_2)
    emit3func = gen_emit(emission3_init, hv_shift, emit3_initphase_2)
    s = (Array(ref_data[:, :, i]) .- ref_mean) 
    s[6,:,:] .=  s[6,:,:] .* β
    X = (F.U[:, 1:n_components]' * s)
    nc =  1.0e-4 # Constant for rate normalization for gradient descent.
    (X = X, t = times,
        U = (u, p, t) -> [emit1func(t), emit2func(t), emit3func(t), hv_func(t)],
        p=[press, tempk, emission1_init, emission2_init, emission3_init, 
        emit1_initphase_2, emit2_initphase_2, emit3_initphase_2,hv_shift]
    )
end

probnames = Tuple(Symbol.(["prob$i" for i in 1:nruns]));
probdata = Tuple([create_prob(ref_data, ref_params, times, i) for i in 1:nruns]);
probtuple = NamedTuple{probnames}(probdata);
probs = DataDrivenDiffEq.ContinuousDataset(probtuple);

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
    #display(prob2.p)
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
    for i ∈ 1:nruns
        estimate = solve_rmse(ode_prob, ref_params[i], probs.probs[i], all_params)
        decoded = (F.U[:, 1:n_components] * Array(estimate))
        decoded[6,:] = decoded[6,:]./β
        decoded = decoded .+ ref_mean
            
        push!(estimates,Array(decoded)) #
    end
    estimates = permutedims(reshape(vcat(estimates...),(nspecs ,nruns,:)),(1,3,2))
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

ref_data
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

function plot_rollouts(ode_prob, ref_params, probs, all_params)
    p = [] 
    for i ∈ 1:30
         push!(p, plot_rollout(ode_prob, ref_params[i], probs.probs[i], all_params;
             legend=(i==1 ? :best : :none),legendfontsize=8))
    end
    plot(p...,size=(2000, 1400))
 end


ref_data_compressed = []
for i in 1:nruns
    push!(ref_data_compressed, probs.probs[i].X)
end
ref_data_compressed = permutedims(reshape(vcat(ref_data_compressed...),(n_components,nruns,:)),(1,3,2)) 


nruns_test = 375
ndays_test = 10

JLD2.@load "../../dataset/ref_data_test.jld" ref_data_test ref_params_test times_test
probnames_test = Tuple(Symbol.(["prob$i" for i in 1:nruns_test]));
probdata_test = Tuple([create_prob(ref_data_test, ref_params_test, times_test, i) for i in 1:nruns_test]);
probtuple_test = NamedTuple{probnames_test}(probdata_test);
probs_test = DataDrivenDiffEq.ContinuousDataset(probtuple_test);


ode_prob_test = ODEProblem(simple_sys, zeros(n_components), (times_test[1], times_test[end]), ps)

import ColorSchemes.grayC100


function rmse_decoded(ode_prob, ref_params, probs, all_params, ref_data, ref_mean, nruns, specs)
    estimates = [] 
    truedata = []
    rmse_sum = 0.0
    for i ∈ 1:nruns
        estimate = solve_rmse(ode_prob, ref_params[i], probs.probs[i], all_params)
        decoded = (F.U[:, 1:n_components] * Array(estimate))
        decoded[6,:] = decoded[6,:]./β
        decoded = decoded .+ ref_mean
        push!(estimates,Array(decoded)) 
    end
    truedata = ref_data[specs,:,:]
    estimates = permutedims(reshape(vcat(estimates...),(nspecs ,nruns,:)),(1,3,2))[specs,:,:]
    #println(size(truedata),size(estimates))
    truedata = reshape(truedata, (size(truedata)[1]*size(truedata)[2]*size(truedata)[3],))
    estimates = reshape(estimates, (size(estimates)[1]*size(estimates)[2]*size(estimates)[3],))
    RMSE = rmsd(truedata, estimates)
end

RMSETestDecoded = rmse_decoded(ode_prob_test, ref_params_test, probs_test, ode_prob_test.p, ref_data_test, ref_mean, nruns_test, 6:6)
push!(RMSEs, RMSETestDecoded)
end

Any[0.06295940864647918, 0.06114453316146538, 0.033951736967373795, 0.042972821700981496]
RMSEs = Float64.(RMSEs)
println(RMSEs)

computetion_times_tsit = []
computetion_times_rb = []
for idx in 1:4
    JLD2.@load "benchmark_$(nruns)_$(idx).jld" bench_ref_simu mean_ref_integ_timesteps bench_sindy_tsit bench_sindy_rb mean_sindy_integ_timesteps_tsit mean_sindy_integ_timesteps_rb
    global ref_simu_time = minimum(bench_ref_simu).time/1e9
    push!(computetion_times_tsit, minimum(bench_sindy_tsit).time/1e9)
    push!(computetion_times_rb, minimum(bench_sindy_rb).time/1e9)
end
computetion_times_tsit
computetion_times_rb
ref_simu_time
computetion_times_tsit = computetion_times_tsit ./ ref_simu_time
computetion_times_rb = computetion_times_rb ./ ref_simu_time
#computetion_times = [4.326, 6.200, 10.882, 25.759]./67.402
#computetion_times = [2.692, 3.140, 5.672, 12.933 ]./53.612
(mean(ref_data_test[6,:,:]))^0.5
sqrt(mean(ref_data_test[6,:,:]))
ref_data_test

pp_time = plot([computetion_times_tsit computetion_times_rb],
marker = (:circle,5),
xlabel = "Latent species",
ylabel = "Computational time \n (SINDy / reference)",
legend=:topright, label = ["Computational time Tsit5" "Rosenbrock23" ],
color = [:blue :red],
titlefontsize = 17,
xtickfontsize = 14, ytickfontsize = 14, 
xlabelfontsize = 16, ylabelfontsize = 16,
legendfontsize = 16,grid = true,
size = (700,450),
left_margin = 3Plots.mm,
right_margin = 5Plots.mm,
bottom_margin = 2Plots.mm)

subplot1 = twinx(pp_time)
plot!(subplot1, [1,2,3,4], [fill(NaN, size(RMSEs)) RMSEs],
        marker = (:circle,5),
        titlefontsize = 17,
        xtickfontsize = 14, ytickfontsize = 14, 
        xlabelfontsize = 16, ylabelfontsize = 16,
        legendfontsize = 16,grid = true,
        legend = :none,
        ylabel = "RMSE",
        color = [:black],
        size = (700,450),
        left_margin = 3Plots.mm,
        right_margin = 5Plots.mm,
        bottom_margin = 2Plots.mm)
pp

savefig(pp_time, "raw/fig6.svg")
#subplot2 = twinx(pp)
#plot!(subplot2, n_components_, [fill(NaN, size(computetion_times)) computetion_times],
#        marker = (:circle,5),
#        legend=:topright, label = ["RMSE" "Computational time"],
#        ylabel = "Computational time (sec)",
#        color = [:blue :red])
#plot_rollouts_decoded(ode_prob_test, times_test, ref_params_test, probs_test, ref_data_test, ode_prob_test.p,6:6)

#savefig(plot_rollouts_decoded(ode_prob_test, times_test, ref_params_test, probs_test, ref_data_test, ode_prob_test.p,6:6),ProjectPath*"plot/fig5_ConcTraj/fig/SINDyTest_Ozone.png") 



