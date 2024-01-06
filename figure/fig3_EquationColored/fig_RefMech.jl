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

ref_model = ChemSys()
nspecs = 11
equations(convert(ODESystem, ref_model.rn))
ndays = 3
timelength = 60 * (ndays * 24) # minutes
saveat = 60.0 # minutes

idx = 0

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

idx = 3
n_components = [1, 2, 3, 4][idx]
λ=[7.408656834939568e-6, 5.074705239490476e-6, 4.010572880855496e-5, 2.7471207892708143e-5][idx]
ε=[8.62723729246145e-5, 5.169241873891593e-6, 2.5049415421745025e-5, 1.0092715146305697e-6][idx]
β=[5.202404809619239, 3.2605210420841684 , 3.1843687374749496, 5.754509018036073][idx]


plot_case_medians = []
global rmsebycases = []
global ref6s = []
global pred6s = []
global median_inds = []
global var6s = ones(4,3)

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

print(h)

JLD2.@load "../../sindy_model/model_system_params_3000_ncomp_$(3).jld" system params F ref_mean

sys_eqn = (equations(system))
nexp = 3
n_components = 3
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

using Latexify
latexify(sys_eqn) |> print

latent_conc = []
for i in 1:nruns
s = (Array(ref_data[:, :, i:i]) .- ref_mean) 
s[6,:,:] .=  s[6,:,:] .* β
s= reshape(s,(size(s)[1],size(s)[2]*size(s)[3]))
X = (F.U[:, 1:n_components]' * s)

push!(latent_conc,reshape(mean(X;dims=2),size(X)[1]))
end

latent_conc = hcat(latent_conc...)
mean(abs.(latent_conc);dims=2)
mean((latent_conc);dims=2)


press_mean = []
tempk_mean = []
emit1_mean = []
emit2_mean = []
emit3_mean = []
hv_mean = []
for i in 1:nruns  
    push!(press_mean,ref_params[i][1])  
    push!(tempk_mean,ref_params[i][2])
    t = 0
    emit1 = 0.95*ref_params[i][3]  + 0.05*ref_params[i][3]*sin(t/(1440/2)*2*pi - ref_params[i][6])
    emit2 = 0.95*ref_params[i][4]  + 0.05*ref_params[i][4]*sin(t/(1440/2)*2*pi - ref_params[i][7])
    emit3 = 0.95*ref_params[i][5]  + 0.05*ref_params[i][5]*sin(t/(1440/2)*2*pi - ref_params[i][8])

    hv = max(sin((t+ref_params[i][9])/720*pi - pi/2),0.0) 
       + max((sin((t+ref_params[i][9])/720*pi - pi/2) + 1.0) / 2.0, 0.0)

    push!(press_mean, ref_params[i][1])  
    push!(tempk_mean, ref_params[i][2])
    push!(emit1_mean, emit1)  
    push!(emit2_mean, emit2)
    push!(emit3_mean, emit3)  
    push!(hv_mean, hv)
end

mean(press_mean)
mean(tempk_mean)
mean(emit1_mean)
mean(emit2_mean)
mean(emit3_mean)
mean(hv_mean)

print(ps)
# File "updated_color.csv" records the values of each term on the right hand side of the surrogate model 
color_csv = CSV.read(pwd()*"/updated_color.csv", DataFrame, header=true)
color_array = Array(color_csv)[:,2:end]
import ColorSchemes.Reds_9
Reds_9[end]
u1 = (abs.(color_array[1:15,1:1]))
heatplot1 = heatmap(u1,c = cgrad([Reds_9[4], :red], 50, categorical = true),title="u1")
Reds_9[4]

u2 =(abs.(color_array[1:19,2:2]))
import ColorSchemes.Blues_9
heatplot2 = heatmap(u2,c = cgrad([Blues_9[4], :blue], 50, categorical = true),title="u2")


u3 = (abs.(color_array[1:20,3:3]))
import ColorSchemes.Greys_9
heatplot3 = heatmap(u3,c = cgrad([Greys_9[4], :black], 50, categorical = true),title="u3")

plot(heatplot1,heatplot2,heatplot3)
savefig(plot(heatplot1,heatplot2,heatplot3),"ucolor.svg")