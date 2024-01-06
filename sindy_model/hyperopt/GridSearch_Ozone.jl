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
using StableRNGs
import ColorSchemes.tab20
#include("../main/ref_model_hvvar.jl")
#using .RefModel

using DiffEqBase, OrdinaryDiffEq
using Catalyst
using Random
using JLD
using Surrogates

function newmodel(press=1.0::Float64, tempk=298.0::Float64, emission=[1.0; 1.0; 1.0; 1.0; 1.0; 1.0]::Vector{Float64}, hv_shift=0.0::Float64)
    # function definition to convert from cm3 molec-1 s-1 to ppm min-1
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

    # hv(t) = abs(ceil(0.1*sin(t/720*pi - pi/2)))
    hv_1(t) = max(sin((t+hv_shift)/720*pi - pi/2),0.0)
    hv_2(t) = max((sin((t+hv_shift)/720*pi - pi/2) + 1.0) / 2.0,0.0)
    hv(t) = hv_1(t) * 0.9 + hv_2(t) * 0.1
    # hv(t) = (mod(t,1440)>720)*1.0
    emit1_initphase = emission[4]
    emit2_initphase = emission[5]
    emit3_initphase = emission[6]

    const1 = emission[1]
    const1_func(t) = 0.95*const1 + 0.05*const1*sin(t/720*pi - emit1_initphase)
    const2 = emission[2]
    const2_func(t) = 0.95*const2 + 0.05*const2*sin(t/720*pi - emit2_initphase)
    const3 = emission[3]
    const3_func(t) = 0.95*const3 + 0.05*const3*sin(t/720*pi - emit3_initphase)
    
    rn = @Catalyst.reaction_network begin
        0.001 * const1_func(t), ∅ → NO2
        0.1 * const2_func(t), ∅ → HCHO
        0.001 * const3_func(t), ∅ → HO2H #CO, NO
        
        0.5 * hv(t), NO2 → NO + O
        rk2, O + O2 → O3
        rk3, O3 + NO → NO2 + O2
        0.015  * hv(t), HCHO + 2O2 → 2HO2 + CO 
        0.022  * hv(t), HCHO → H2 + CO
        rk6, HCHO + HO + O2 → HO2 + CO + H2O
        rk7, HO2 + NO → HO + NO2
        rk8, HO + NO2 → HNO3
        0.0003  * hv(t), HO2H → 2HO
        rk10, HO2H + HO → H2O + HO2
        
        0.4/(100*12) * 60, O3 --> ∅; # Hauglustaine et al. 1994
        4/(100*15)* 60, HNO3 --> ∅;
        # 0.5/(100*2) * 60, HO2H --> ∅;
        0.5/(100*2) * 60, HO2 --> ∅;
        0.1/(100*2) * 60, NO2 --> ∅;
        0.03/2 * 60, CO --> ∅;
        0.1/100 * 60, H2 --> ∅; # Ehhalt and Rohrer, 2009     avg conc 0.4~0.6 ppm

    end # rk2 rk3 rk6 rk7 rk8 rk10

    p = [rk2(press,tempk), rk3(press,tempk), rk6(press,tempk), 
        rk7(press,tempk), rk8(press,tempk), rk10(press,tempk)]

    nspecs = size(species(rn),1)

    specmap = Dict()
    for (k, v) in speciesmap(rn)
        specmap[replace(string(k), r"\(t\)$"=>"")] = v # remove '(t)' from names
    end
    natoms = 4 # C H O N
    atoms = zeros(Float64, nspecs, natoms)
    atoms[specmap["O3"], :] = [0 0 3 0]
    atoms[specmap["NO"], :] = [0 0 1 1]
    atoms[specmap["NO2"], :] = [0 0 2 1]
    atoms[specmap["HCHO"], :] = [1 2 1 0]
    atoms[specmap["HO2"], :] = [0 1 2 0]
    atoms[specmap["HO2H"], :] = [0 2 2 0]
    atoms[specmap["HO"], :] = [0 1 1 0]
    atoms[specmap["O"], :] = [0 0 1 0]
    atoms[specmap["HNO3"], :] = [0 1 3 1]
    atoms[specmap["CO"], :] = [1 0 1 0]
    atoms[specmap["H2"], :] = [0 2 0 0]
    atoms[specmap["H2O"], :] = [0 2 1 0]
    atoms[specmap["O2"], :] = [0 0 2 0]

    rn, atoms, specmap, p
end

struct ChemSys
    rn
    atoms::Array{Float64, 2}
    specmap::Dict
    p::Array{Float64, 1}
end

# Create a reference model at the given temperature and pressure.
ChemSys(press=1.0::Real, tempk=298.0::Real, emission=[1.0; 1.0; 1.0; 1.0; 1.0; 1.0]::Vector{Float64}, hv_shift=0.0::Float64) = ChemSys(newmodel(press, tempk, emission, hv_shift)...)

# run the specified number of reference model simulations for the given timelength,
# and return results spaced at the given time step length,
# both in minutes.
function random_sims(m::ChemSys, nruns::Int, minutes, saveat, idx)
    c0 = zeros(Float64,length(m.specmap))
        c0[m.specmap["O3"], :] .= 0.001*10^(2*rand())         # 03 range: 0.001 - 0.1 ppm
        c0[m.specmap["NO"], :] .= 0.0015*10^(2*rand())        # NO range: 0.0015 - 0.15
        c0[m.specmap["NO2"], :] .= 0.0015*10^(2*rand())        # NO2 range: 0.0015 - 0.15
        c0[m.specmap["HCHO"], :] .= 0.02*10^(2*rand())          # HCHO range: 0.02 - 2 ppm
        c0[m.specmap["HO2"], :] .= 1.0e-05 * rand()            # HO2. range: 1 - 10 *ppt*
        c0[m.specmap["HO2H"], :] .= 0.01*rand()	               # HO2H range: 0.001 - 0.01 ppm
        #c[7:11] = zeros(Float64,1,5)
    c0[m.specmap["O2"], :] .= 0.21e6	               # O2: 0.21e6 ppm

    tspan = (0, minutes) # minutes
    prob = ODEProblem(m.rn, c0, tspan, m.p)
    # 
    
    # press, tempk, emission1, emission2, emission3, hv_shift
    p_lower = [0.9, 288.0, 0.5, 0.5, 0.5, 0.0 ,0.0, 0.0, 0.0]
    p_upper = [1.1, 308.0, 1.5, 1.5, 1.5, 2*pi , 2*pi, 2*pi, 1440.0]
    # select set of Sobol-sampled parameter vectors
    #Random.seed!(42)
    para = Surrogates.sample(1000000, p_lower, p_upper, SobolSample())[nruns * idx+1 : nruns * idx+nruns]
    #println(para)
    m = ChemSys(para[1][1], para[1][2], collect(para[1][3:8]), para[1][9])

    return m
end

ref_model = ChemSys()
nspecs = 11

#equations(convert(ODESystem, ref_model.rn))
DataPath =""
cd(Base.source_path()*"/..")

saveat = 60.0 # minutes

nruns_train = 3000
ndays_train = 3
timelength_train = 60 * (ndays_train * 24) # minutes
JLD2.@load DataPath*"../dataset/ref_data_train.jld" ref_data_train ref_params_train times_train
ref_data_train

want_specs = setdiff(1:13, [ref_model.specmap["O2"], ref_model.specmap["H2O"]])
specs = reshape((states(ref_model.rn)[want_specs]),(1,11))
specs = [string(i)[1:(end-3)] for i in specs ]

nruns_validate = 375
ndays_validate = 10
timelength_validate = 60 * (ndays_validate * 24)

JLD2.@load DataPath*"ref_data_validate.jld" ref_data_validate ref_params_validate times_validate


    
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





####################################################################################
function CtrlExpr(λ,  ε, enhance_ozone,n_components)

#n_components = 3
ref_std_train = 1.0#std(Array(ref_data), dims=(2,3))[:]
ref_mean_train = mean(Array(ref_data_train), dims=(2,3))[:]
B = reshape((Array(ref_data_train) .- ref_mean_train)./ref_std_train , nspecs, :)
B_ = copy(B)
B_[6,:] .=  B_[6,:].*enhance_ozone
B = B_
F = svd(B)

function create_prob(ref_data, ref_params, times, i)
    press, tempk, emission1_init, emission2_init, emission3_init, emit1_initphase_2, emit2_initphase_2, emit3_initphase_2, hv_shift = ref_params[i]
    hv_func = gen_hv(hv_shift)
    emit1func = gen_emit(emission1_init, hv_shift, emit1_initphase_2)
    emit2func = gen_emit(emission2_init, hv_shift, emit2_initphase_2)
    emit3func = gen_emit(emission3_init, hv_shift, emit3_initphase_2)
    s = (Array(ref_data[:, :, i]) .- ref_mean_train) ./ ref_std_train
    s[6,:,:] .=  s[6,:,:] .* enhance_ozone
    X = (F.U[:, 1:n_components]' * s)
    nc =  1.0e-4 # Constant for rate normalization for gradient descent.
    (X = X, t = times,
        U = (u, p, t) -> [emit1func(t), emit2func(t), emit3func(t), hv_func(t)],
        p=[press, tempk, emission1_init, emission2_init, emission3_init, 
        emit1_initphase_2, emit2_initphase_2, emit3_initphase_2,hv_shift]
    )
end

probnames_train = Tuple(Symbol.(["prob$i" for i in 1:nruns_train]));
probdata_train = Tuple([create_prob(ref_data_train, ref_params_train, times_train, i) for i in 1:nruns_train]);
probtuple_train= NamedTuple{probnames_train}(probdata_train);
probs_train = DataDrivenDiffEq.ContinuousDataset(probtuple_train);


function solve_rmse(ode_prob, forcing_params, data_prob, all_params; kwargs...)
    prob2 = prob_func(ode_prob, forcing_params, all_params, data_prob.X[:,1])
    estimate = solve(prob2, Rosenbrock23(), saveat=saveat)
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
    for i ∈ 1:nruns
        estimate = solve_rmse(ode_prob, ref_params[i], probs.probs[i], all_params)
        decoded = (F.U[:, 1:n_components] * Array(estimate))
        decoded[6,:] = decoded[6,:]./enhance_ozone
        decoded = decoded .+ ref_mean_train
    
        #push!(truedata,Array(truedata_))#
            
        push!(estimates,Array(decoded)) #
    end
    #truedata = permutedims(reshape(vcat(truedata...),(nspecs ,nruns,:)),(1,3,2)) 
    estimates = permutedims(reshape(vcat(estimates...),(nspecs ,nruns,:)),(1,3,2))#
    #RMSE = rmsd(truedata, estimates)
end


function rmse_decoded(ode_prob, ref_params, probs, all_params, ref_data, ref_mean, nruns, specs)
    estimates = [] 
    truedata = []
    rmse_sum = 0.0
    for i ∈ 1:nruns
        estimate = solve_rmse(ode_prob, ref_params[i], probs.probs[i], all_params)
        
        #truedata_ = (F.U[:, 1:n_components] * Array(probs.probs[i].X)) .+ ref_mean
        decoded = (F.U[:, 1:n_components] * Array(estimate))
        decoded[6,:] = decoded[6,:]./enhance_ozone
        #println(size(decoded))
        #println(size(ref_mean))
        decoded = decoded .+ ref_mean
    
        #push!(truedata,Array(truedata_))#
            
        push!(estimates,Array(decoded)) #
    end
    truedata = ref_data[specs,:,:]
    estimates = permutedims(reshape(vcat(estimates...),(nspecs ,nruns,:)),(1,3,2))[specs,:,:]
    println(size(truedata),size(estimates))
    truedata = reshape(truedata, (size(truedata)[1]*size(truedata)[2]*size(truedata)[3],))
    estimates = reshape(estimates, (size(estimates)[1]*size(estimates)[2]*size(estimates)[3],))
    #println(size(truedata),size(estimates))
    RMSE = rmsd(truedata, estimates)
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


#for loop in 1:nloop
#################### Part 2: Use SINDy to find the model equations ####################
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
    #fourier_basis(u, 2);
]



#begin # Run SINDy a bunch of times because it doesn't always work well. Keep the best result.

basis = Basis(h, u, parameters = [press, tempk, 
emission1_init, emission2_init, emission3_init, emit1_initphase_2, 
emit2_initphase_2, emit3_initphase_2, hv_shift], 
controls = [emission1, emission2, emission3, hv])
opt = STLSQ(λ)

tmp_res = solve(probs_train, basis, opt)
global rss_val = rss(tmp_res)
global res = tmp_res



system = get_basis(res)
params = get_parameter_map(system)

global sys_eqn = (equations(system))
nexp = 4
if nexp%2==0
    expo = nexp+1
else
    expo = nexp +2
end

for i in 1:n_components
    #sys_eqn = @set sys_eqn[i].rhs +=  - ε*(u[i])^nexp
end
#using Latexify
#latexify(sys_eqn) |> display  or print



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

   
probnames_validate = Tuple(Symbol.(["prob$i" for i in 1:nruns_validate]));
probdata_validate = Tuple([create_prob(ref_data_validate, ref_params_validate, times_validate, i) for i in 1:nruns_validate]);
probtuple_validate = NamedTuple{probnames_validate}(probdata_validate);
probs_validate = DataDrivenDiffEq.ContinuousDataset(probtuple_validate);

ode_prob_validate = ODEProblem(simple_sys, zeros(n_components), (times_validate[1], times_validate[end]), ps)
try
    RMSETestDecoded = rmse_decoded(ode_prob_validate, ref_params_validate, probs_validate, ode_prob_validate.p, ref_data_validate, ref_mean_train, nruns_validate, 1:11)
catch e
    println(e)
    RMSETestDecoded = 10000.0
end 

end

###################################################################################
n_components = parse(Int64, ARGS[1])
println("n_components = $(n_components)")
n_gs = 1000
if n_components > 4
    n_gs = 200
end

ho = @hyperopt for i = n_gs,
    sampler = LHSampler(),          # LHSampler(),CLHSampler(), Hyperband()
    λ = exp10.(LinRange(-4,-6,n_gs)),   # SINDy threshold
    ε = exp10.(LinRange(-4,-6,n_gs)),   # ODE RHS buffer weight
    enhance_ozone = (LinRange(0.5,10.0,n_gs))
    #n_components = [n_component]

print(i, " \t", "λ=$(λ)", " \t", "ε=$(ε)", " \t", "EO=$(enhance_ozone)", " \t", "n_components=$(n_components)","\n")
@show CtrlExpr(λ, ε, enhance_ozone, n_components)
end
best_params, min_f = ho.minimizer, ho.minimum
@time CtrlExpr(best_params[1], best_params[2], best_params[3], best_params[4])
println(best_params, min_f )
printmin(ho)

