γ(x, μ, c) = gammahaz(μ, c, x)
d(x, t, μ, c, Ton, Toff) = mod(t, Ton + Toff) < Ton ? gammahaz(μ, c, x) : 0.0

@register_symbolic γ(x, μ, c)
@register_symbolic d(x, t, μ, c, Ton, Toff)

@independent_variables t 
@variables x(t)
@parameters pγ[1:2] pd[1:2] pT[1:2]
psyms = [pγ, pd, pT]

sys = complete(ODESystem(Equation[], t, [x,], psyms; name=:sys))

matLsymb = [
    -γ(x, pγ...) - d(x, t+x, pd..., pT...)  0.0;
    0.0 0.0]

matLinvsymn = simplify.(inv(matLsymb))

matMsymb = [
    2*γ(x, pγ...) 0;
    0 0]
param_map = Dict(
    pγ => [1.8, 1.0],
    pd => [2.0, 1e-3],
    pT => [T1_, T1_])

vary_T = exp.(range(-1.0, stop=2.5, length=20)) 
main(matMsymb, matLsymb, sys; 
    param_map=param_map, trange=vary_T, iters=100, xintsteps=1000, tintsteps=1000, name="nodorm")

plot_lambda(;iters=100, xintsteps=1000, name="nodorm")
