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

function random_sims_restart(m::ChemSys, nruns::Int, minutes, idx, c0_, param_)

    c0 = c0_
    tspan = (0, minutes) # minutes
    prob = ODEProblem(m.rn, c0[:,1], tspan, param_[1])
    # 
    
    # press, tempk, emission1, emission2, emission3, hv_shift
    #p_lower = [0.9, 288.0, 0.5, 0.5, 0.5, 0.0 ,0.0, 0.0, 0.0]
    #p_upper = [1.1, 308.0, 1.5, 1.5, 1.5, 2*pi , 2*pi, 2*pi, 1440.0]
    # select set of Sobol-sampled parameter vectors
    #Random.seed!(42)
    #para = Surrogates.sample(3750, p_lower, p_upper, SobolSample())[idx]
    
    
    function prob_func(prob,i,repeat)
        m = ChemSys(param_[i][1], param_[i][2], collect(param_[i][3:8]), param_[i][9])
        prob = ODEProblem(m.rn, c0[:,i], tspan, m.p)
    end
    
    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
    
    res = solve(ensemble_prob, Rosenbrock23(), trajectories=nruns, maxiters=Int(1e10), progress=true)
    return  res, param_
end

ref_model = ChemSys()
nspecs = 11
equations(convert(ODESystem, ref_model.rn))

ndays = 3
timelength = 60 * (ndays * 24) # minutes
timelength_restart = timelength - Int(timelength /3 + 60) 
saveat = 60.0 # minutes
nruns = 3000
idx = 0

ref_res, para_ = random_sims(ref_model, nruns, timelength, saveat, idx)
ref_res
for j in 1:10
    for i in 1:length(ref_res)
        if ref_res[i].retcode != :Success
            ref_res[i] = ref_res[i-1]
        end
    end
end

istep_restart = Int((timelength /3)/saveat+2)
ref_res[:,istep_restart,:]

bench_ref_simu = @benchmark ref_res_2d, param_2d = random_sims_restart(ref_model, nruns, timelength_restart, idx, ref_res[:,istep_restart,:], para_)
ref_res_2d, param_2d = random_sims_restart(ref_model, nruns, timelength_restart, idx, ref_res[:,istep_restart,:], para_)
println("bench_ref_simu = $(bench_ref_simu)")
ref_res_2d
for j in 1:10
    for i in 1:length(ref_res_2d)
        if ref_res_2d[i].retcode != :Success
            ref_res_2d[i] = ref_res_2d[i-1]
        end
    end
end


ref_integ_timesteps = []
for i in 1:nruns
    ref_integ_timestep = length(ref_res_2d[i].t)
    push!(ref_integ_timesteps, ref_integ_timestep)
end
ref_integ_timesteps
mean_ref_integ_timesteps = mean(ref_integ_timesteps)


ref_data_all = []
ref_params_all = []
JLD2.@load "../../dataset/ref_data_train.jld" ref_data_train ref_params_train times_train
push!(ref_data_all, ref_data_train)
push!(ref_params_all, ref_params_train)

global times = times_train
global ref_data = cat(ref_data_all...;dims=3)
global ref_params = vcat(ref_params_all...)

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


for idx in 1:1
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
global probs = DataDrivenDiffEq.ContinuousDataset(probtuple);


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

JLD2.@load "../../sindy_model/model_system_params_$(3000)_ncomp_$(n_components).jld" system params F ref_mean

sys_eqn = (equations(system))
nexp = 3
if nexp%2==0
    expo = nexp+1
else
    expo = nexp +2
end

for i in 1:n_components
    sys_eqn = @set sys_eqn[i].rhs +=  - ε*(u[i])^nexp
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
global ode_prob = ODEProblem(simple_sys, zeros(n_components), (times[1], times[end]),ps)

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


function solve_func_return_timestep(ode_prob, forcing_params, data_prob, all_params; kwargs...)
    prob2 = prob_func(ode_prob, forcing_params, all_params, data_prob.X[:,1])
    estimate = solve(prob2, Rosenbrock23())
    length(estimate.t)
end


function solve_sindy_return_timestep(ode_prob, ref_params, probs, all_params, nruns)
    estimate_timesteps = [] 
    for i ∈ 1:nruns
        estimate_timestep = solve_func_return_timestep(ode_prob, ref_params[i], probs.probs[i], all_params)
        push!(estimate_timesteps,estimate_timestep) #
    end
    estimate_timesteps
end


global function run_ensemble(prob, forcing_params, data_probs, all_params)
    function setup(prob,i,repeat)
        u0 = data_probs[i].X[:,1]
        prob_func(prob, forcing_params[i], all_params, u0)
    end

    global ensemble_prob = EnsembleProblem(prob, prob_func=setup)

    bench_sindy =  solve(ensemble_prob, Rosenbrock23(),  trajectories=nruns, 
        saveat=saveat)
    return bench_sindy
end


bench_sindy = @benchmark run_ensemble(ode_prob, ref_params, probs.probs, ode_prob.p)


est = solve_sindy_return_timestep(ode_prob, ref_params, probs, ode_prob.p, nruns)
mean_sindy_integ_timesteps = mean(est)

println("n_comp = $(idx)")
println("bench_ref_simu = $(bench_ref_simu)")
println("mean_ref_integ_timesteps = $(mean_ref_integ_timesteps)")
println("bench_sindy = $(bench_sindy)")
println("mean_sindy_integ_timesteps = $(mean_sindy_integ_timesteps)")

JLD2.jldsave("benchmark_$(nruns)_$(idx).jld"; bench_ref_simu, mean_ref_integ_timesteps, bench_sindy, mean_sindy_integ_timesteps)
end

for idx in 1:4
    JLD2.@load "benchmark_$(nruns)_$(idx).jld" bench_ref_simu mean_ref_integ_timesteps bench_sindy mean_sindy_integ_timesteps
    println("n_comp = $(idx)")
    println("bench_ref_simu = $(bench_ref_simu)")
    println("mean_ref_integ_timesteps = $(mean_ref_integ_timesteps)")
    println("bench_sindy = $(bench_sindy)")
    println("mean_sindy_integ_timesteps = $(mean_sindy_integ_timesteps)")
end
