include("compute.jl")
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "--gamma"
        arg_type = Float64
        nargs = 2 
        default = [1.8, 1.0]
    "--d"
        arg_type = Float64
        nargs = 2
        default = [2.0, 1e-3]
    "--on"
        arg_type = Float64
        nargs = 2 
        default = [0.1, 1.0]
    "--off"
        arg_type = Float64
        nargs = 2 
        default = [0.1, 1.0]
    "--Trange"
        arg_type = Float64
        nargs = 2 
        default = [0.0, 1.5]
    "--tsteps"
        arg_type = Int 
        default = 20
    "--xintsteps"
        arg_type = Int 
        default = 1000
    "--tintsteps"
        arg_type = Int 
        default = 1000
    "--maxiters"
        arg_type = Int 
        default = 1000
    "--name"
        default = "dormancy"
end

parsed_args = parse_args(ARGS, s)

γ(x, μ, c) = gammahaz(μ, c, x)
d(x, t, μ, c, Ton, Toff) = mod(t, Ton + Toff) < Ton ? gammahaz(μ, c, x) : 0.0
τoff(x, t, μ, c, Ton, Toff) = mod(t, Ton + Toff) < Ton ? gammahaz(μ, c, x) : 0.0
τon(x, t, μ, c, Ton, Toff) = mod(t, Ton + Toff) < Ton ? 0.0 : gammahaz(μ, c, x) 

@register_symbolic γ(x, μ, c)
@register_symbolic d(x, t, μ, c, Ton, Toff)
@register_symbolic τoff(x, t, μ, c, Ton, Toff)
@register_symbolic τon(x, t, μ, c, Ton, Toff)

@independent_variables t 
@variables x(t)
@parameters pγ[1:2] pd[1:2] pτon[1:2] pτoff[1:2] pT[1:2] 
psyms = [pγ, pd, pτon, pτoff, pT]

sys = complete(ODESystem(Equation[], t, [x,], psyms; name=:sys))

matLsymb = [
    -γ(x, pγ...) - d(x, t+x, pd..., pT...) - τoff(x, t+x, pτoff..., pT...) 0.0;
    0.0 -τon(x, t+x, pτon..., pT...)]

matLinvsymn = simplify.(inv(matLsymb))

matMsymb = [
    2*γ(x, pγ...) τon(x, t, pτon..., pT...);
    τoff(x, t, pτoff..., pT...) 0]

param_map = Dict(
    pγ => parsed_args["gamma"],
    pd => parsed_args["d"],
    pτoff => parsed_args["off"], 
    pτon => parsed_args["on"], 
    pT => [0.1, 0.1])

#param_map = Dict(
#    pγ => [1.8, 1.0],
#    pd => [2.0, 1e-3],
#    pτoff => [0.1, 1.0], 
#    pτon => [0.1, 1.0], 
#    pT => [0.1, 0.1])

tspan = parsed_args["Trange"]
vary_T = exp.(range(tspan[1], stop=tspan[2], length=parsed_args["tsteps"])) 
main(matMsymb, matLsymb, sys; 
     param_map=param_map, 
     trange=vary_T, 
     iters=parsed_args["maxiters"], 
     xintsteps=parsed_args["xintsteps"], 
     tintsteps=parsed_args["tintsteps"], 
     name=parsed_args["name"])
