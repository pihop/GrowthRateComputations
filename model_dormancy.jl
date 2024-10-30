include("compute.jl")

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
@parameters pγ[1:2] pd[1:2] pτon[1:2] pτoff[1:2] pT[1:2] T1
psyms = [pγ, pd, pτon, pτoff, pT, T1]

sys = complete(ODESystem(Equation[], t, [x,], psyms; name=:sys))

matLsymb = [
    -γ(x, pγ...) - d(x, t+x, pd..., pT...) - τoff(x, t+x, pτoff..., pT...) 0.0;
    0.0 -τon(x, t+x, pτon..., pT...)]

matLinvsymn = simplify.(inv(matLsymb))

matMsymb = [
    2*γ(x, pγ...) τon(x, t, pτon..., pT...);
    τoff(x, t, pτoff..., pT...) 0]

param_map = Dict(
    pγ => [1.8, 1.0],
    pd => [2.0, 1e-3],
    pτoff => [0.1, 1.0], 
    pτon => [0.1, 1.0], 
    pT => [0.1, 0.1])

vary_T = exp.(range(0.0, stop=1.5, length=20)) 
main(matMsymb, matLsymb, sys; 
    param_map=param_map, trange=vary_T, iters=100, xintsteps=1000, tintsteps=1000, name="dormancy")
