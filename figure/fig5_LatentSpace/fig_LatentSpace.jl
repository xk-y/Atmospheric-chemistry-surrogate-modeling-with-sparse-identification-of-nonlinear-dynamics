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


# Functions related to reference model.
#begin
    
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
#end


idx = 3
n_components = [1, 2, 3, 4][idx]
λ=[7.408656834939568e-6, 5.074705239490476e-6, 4.010572880855496e-5, 2.7471207892708143e-5][idx]
ε=[8.62723729246145e-5, 5.169241873891593e-6, 2.5049415421745025e-5, 1.0092715146305697e-6][idx]
β=[5.202404809619239, 3.2605210420841684 , 3.1843687374749496, 5.754509018036073][idx]


ref_mean = mean(Array(ref_data), dims=(2,3))[:]
ref_std = 1.0
B = reshape((Array(ref_data) .- ref_mean)./1.0 , nspecs, :)
B_ = copy(B)
B_[6,:] .=  B_[6,:].*β
#B_[3,:] .=  B_[3,:].*β
B = B_
F = svd(B)
plot_SVD = plot(-(F.S.^2 ./ sum(F.S.^2)).+1.0, 
    xlabel="Latent species",
    ylabel="Total variance unexplained",
    #title="Singular value contributions",
    xticks = 0:1:11,
    yticks = 0.0:0.1:1.0,
    xlims = (1,11),
    ylims = (0.3,1.01),
    xtickfontsize = 20, ytickfontsize = 20, 
    xlabelfontsize = 23, ylabelfontsize = 23,
    label = :none,
    linewidth = 4,
    color = :red,
    bottom_margin = -1Plots.mm,
    right_margin = 4Plots.mm,
    size = (600,630))

savefig(plot_SVD, "SVD.svg")



function create_prob(ref_data, ref_params, times, i)
    press, tempk, emission1_init, emission2_init, emission3_init, emit1_initphase_2, emit2_initphase_2, emit3_initphase_2, hv_shift = ref_params[i]
    hv_func = gen_hv(hv_shift)
    emit1func = gen_emit(emission1_init, hv_shift, emit1_initphase_2)
    emit2func = gen_emit(emission2_init, hv_shift, emit2_initphase_2)
    emit3func = gen_emit(emission3_init, hv_shift, emit3_initphase_2)
    s = (Array(ref_data[:, :, i]) .- ref_mean) ./ ref_std
    s[6,:,:] .=  s[6,:,:] .* β
    X = (F.U[:, 1:n_components]' * s)
    nc =  1.0e-4 # Constant for rate normalization for gradient descent.
    (X = X, t = times,
        U = (u, p, t) -> [emit1func(t), emit2func(t), emit3func(t), hv_func(t)],
        p=[press, tempk, emission1_init, emission2_init, emission3_init, 
        emit1_initphase_2, emit2_initphase_2, emit3_initphase_2,hv_shift]
    )
end


want_specs = setdiff(1:13, [ref_model.specmap["O2"], ref_model.specmap["H2O"]])
specs = reshape((states(ref_model.rn)[want_specs]),(1,11))
specs = [string(i)[1:(end-3)] for i in specs ]

specs = reshape((states(ref_model.rn)[want_specs]),(1,11))
specs_ = specs
specs_ = [string(i)[1:(end-3)] for i in specs ]
specs_[3] = "H2O2"
#n_components = 11
a = F.U[:, 1:nspecs]
gr()
heatplot = heatmap(a,
                   xticks=(1:nspecs),yticks=(1:11,specs_),
                   clim = (-1,1), 
                   xtickfontsize = 20, ytickfontsize = 20, 
                   colorbar_fontsize = 9,
                   xlabelfontsize = 23, ylabelfontsize = 23,
                   xlabel="Latent species", ylabel="", 
                   right_margin = 15Plots.mm,
                   c = :RdBu_9, 
                   size=(800,630)
                   )
fontsize = 10
nrow, ncol = size(a)
ann = [(i,j, text(round(a[i,j], digits=2), fontsize, :white, :center))
            for i in 1:nrow for j in 1:ncol]
#annotate!(ann, linecolor=:white)
annotate!([(j, i, text(round(a[i,j],digits=3), fontsize, :white)) for i in 1:nrow for j in 1:ncol])
savefig(heatplot, "heatplot.svg")




#probnames = Tuple(Symbol.(["prob$i" for i in 1:nruns]));
#probdata = Tuple([create_prob(ref_data, ref_params, times, i) for i in 1:nruns]);
#probtuple = NamedTuple{probnames}(probdata);
#probs = DataDrivenDiffEq.ContinuousDataset(probtuple);







function plot_PCA(irun, ispec)
    gr()
    plot_ref = plot(times, ref_data[ispec, :,irun]', 
        label=specs[:,ispec], 
        xlabel="Time (s)", ylabel="Conc. (ppm)", 
        palette=:Paired_11, 
        title="Reference data (run $(irun))")
    plot!(times, gen_hv(ref_params[irun][9]).(times), 
        label="hv", 
        color=:grey80)
    
    plot_norm = plot(times, B[ispec, (length(times)*(irun-1)+1):(length(times)*(irun))]', 
        palette=:Paired_11, 
        label=specs[:,ispec], 
        xlabel="Time (s)",ylabel="Conc. (ppm)", 
        title="Normalized data (run $(irun))")
    plot!(times, gen_hv(ref_params[irun][9]).(times), 
        label="hv", 
        color=:grey80)
    
    plot_pca = plot(probs.probs[irun].t, probs.probs[irun].X', 
        palette=:Paired_11, 
        title="PCA data (run $(irun))",
        xlabel="Time (s)", ylabel="Conc. (ppm)")

    plot_emit = plot(times, gen_emit(ref_params[irun][3],ref_params[irun][9],ref_params[irun][6]).(times), 
        xlabel="Time (s)", ylabel="Conc. (ppm)", 
        label="NO2",
        color=:red, 
        title="Emission (run $(irun))")
    
    plot!(times, gen_emit(ref_params[irun][4],ref_params[irun][9],ref_params[irun][7]).(times), 
        label="HCHO",
        color=:blue)
    plot!(times, gen_emit(ref_params[irun][5],ref_params[irun][9],ref_params[irun][8]).(times), 
        label="HO2H",
        color=:black)
    plot!(times, gen_hv(ref_params[irun][9]).(times), 
        label="hv", 
        color=:grey80)
        
        #plot_press = plot(times, times./times.*(ref_params[1][1]), label="press",color=:green, title="Pressure data (run 1)")
        #plot_tempk = plot(times, times./times.*(ref_params[1][2]), label="tempk",color=:orange, title="Temperature data (run 1)")
    plot_PCA = plot(plot_ref, plot_pca, plot_norm, plot_emit, 
        size = (800,600),
        legend = :outerright,
        margin = 10Plots.mm)
end

#savefig(plot_PCA(1,1:11),"PCA_1.svg")


