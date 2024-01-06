module RefModel

using DiffEqBase, OrdinaryDiffEq
using Catalyst
using Random
using JLD
using Surrogates
export ChemSys, random_sims, random_sim

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

    end  #rk2 rk3 rk6 rk7 rk8 rk10

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
    para = Surrogates.sample(3750, p_lower, p_upper, SobolSample())[nruns * idx+1 : nruns * idx+nruns]
    #println(para)

    function prob_func(prob,i,repeat)
        # prob.u0[:] = c0[:, i]
        #press = 0.95 + rand() * 0.1                 # 0.9  -  1.1
        #tempk = 298.0 + rand() * 20 - 10           # 288  -  308
        #emission = 0.2 .+ rand(3) * 1.8              # 0.2  -  2
        #hv_shift = rand()*1440                        # 0 - 1440
        # println("emission\n")
        m = ChemSys(para[i][1], para[i][2], collect(para[i][3:8]), para[i][9])
        prob = ODEProblem(m.rn, c0, tspan, m.p)
    end

    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)

    res = solve(ensemble_prob, Rosenbrock23(), trajectories=nruns, saveat=saveat,maxiters=Int(1e10), progress=true)
    return res, para
end

end
