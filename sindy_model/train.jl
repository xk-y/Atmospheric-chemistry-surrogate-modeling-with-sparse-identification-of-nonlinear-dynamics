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
pwd()
include("../ref_model/ref_model_hvvar.jl")
using .RefModel

#################### Part 1: generate reference data ####################

ref_model = ChemSys()
nspecs = 11

equations(convert(ODESystem, ref_model.rn))

ndays = 3
timelength = 60 * (ndays * 24) # minutes
saveat = 60.0 # minutes
nruns = 3000

JLD2.@load "../../dataset/ref_data_train.jld" ref_data_train ref_params_train times_train

  


ref_data = ref_data_train
ref_params = ref_params_train
times = times_train

    
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


for idx in 1:4
n_components = [1, 2, 3, 4][idx]
λ=[9.637578663841089e-5, 5.3389383971473615e-5, 4.691834510607799e-5, 4.525386278170167e-5][idx]
ε=[2.481930289186253e-5, 1.1644806183726854e-5, 5.991769669310613e-6, 1.400058382468099e-6][idx]
β=[5.031062124248497, 2.8607214428857715 , 3.43186372745491, 8.564064064064064][idx]

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

opt = STLSQ(λ)
tmp_res = solve(probs, basis, opt)

global rss_val = rss(tmp_res)
global res = tmp_res

system = get_basis(res)
params = get_parameter_map(system)

JLD2.jldsave("../model/model_system_params_$(nruns)_ncomp_$(n_components).jld"; system, params, F, ref_mean)

end

pwd()
