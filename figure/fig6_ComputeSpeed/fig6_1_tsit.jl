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



ntimesteps = [31.329666666666668, 70.963, 81.73433333333334, 105.90066666666667]
nterms = [3,21, 60, 126]
prod = Float64.(ntimesteps.*nterms)

computetion_times_tsit = []
#computetion_times_rb = []
for idx in 1:4
    JLD2.@load "benchmark_$(nruns)_$(idx).jld" bench_ref_simu mean_ref_integ_timesteps bench_sindy_tsit bench_sindy_rb mean_sindy_integ_timesteps_tsit mean_sindy_integ_timesteps_rb
    global ref_simu_time = minimum(bench_ref_simu).time/1e9
    push!(computetion_times_tsit, minimum(bench_sindy_tsit).time/1e9)
#    push!(computetion_times_rb, minimum(bench_sindy_rb).time/1e9)
end

pp = plot([1,2,3,4], prod,
        marker = (:circle,5),
        legend = false,
        xlabel = "Latent species",
        ylabel = "Product",
        color = :blue,
        titlefontsize = 17,
        xtickfontsize = 14, ytickfontsize = 14, 
        xlabelfontsize = 16, ylabelfontsize = 16,
        legendfontsize = 16,grid = true,
        size = (700,450),
        left_margin = 3Plots.mm,
        right_margin = 5Plots.mm,
        bottom_margin = 2Plots.mm)
subplot1 = twinx(pp)
plot!(subplot1, [1,2,3,4], Float64.([fill(NaN, size(computetion_times_tsit)) computetion_times_tsit]),
        marker = (:circle,5),
        legend=:topright, label = ["Product" "Computational time"],
        ylabel = "Computational time \n (SINDy / reference)",
        titlefontsize = 17,
        xtickfontsize = 14, ytickfontsize = 14, 
        xlabelfontsize = 16, ylabelfontsize = 16,
        legendfontsize = 16,grid = true,
        size = (700,450),
        left_margin = 3Plots.mm,
        right_margin = 5Plots.mm,
        bottom_margin = 2Plots.mm,
        color = [:blue :red])
pp
savefig(pp, "raw/rosenbrock/fig6_prod_time.svg")

#################################################
pp2 = plot([1,2,3,4], nterms,
        marker = (:circle,5),
        legend = false,
        xlabel = "Latent species",
        ylabel = "Equation terms",
        color = :blue,
        titlefontsize = 17,
        xtickfontsize = 14, ytickfontsize = 14, 
        xlabelfontsize = 16, ylabelfontsize = 16,
        legendfontsize = 16,grid = true,
        size = (700,450),
        left_margin = 3Plots.mm,
        right_margin = 5Plots.mm,
        bottom_margin = 2Plots.mm)
subplot2 = twinx(pp2)
plot!(subplot2, [1,2,3,4], Float64.([fill(NaN, size(ntimesteps)) ntimesteps]),
        marker = (:circle,5),
        legend=:topright, label = ["Equation terms" "Integration timesteps"],
        ylabel = "Integration timesteps",
        titlefontsize = 17,
        xtickfontsize = 14, ytickfontsize = 14, 
        xlabelfontsize = 16, ylabelfontsize = 16,
        legendfontsize = 16,grid = true,
        size = (700,450),
        left_margin = 3Plots.mm,
        right_margin = 5Plots.mm,
        bottom_margin = 2Plots.mm,
        color = [:blue :red])
pp2
savefig(pp2, "raw/rosenbrock/fig6_eqn_time.svg")