using OrdinaryDiffEq
using Roots
using Distributions
using LinearAlgebra
using StatsBase
using Interpolations
using Symbolics
using ModelingToolkit
using Integrals
using CSV
using DataFrames

using CairoMakie
using ColorSchemes
colors = ColorSchemes.Egypt.colors

mutable struct LSolution
    problem
    sol::Dict
    function LSolution(problem, t)
        prob = problem(t) 
        sol = Dict()
        sol[t] = solve(prob, Tsit5()) 
        new(problem, sol)
    end
end

function new_solution!(sol::LSolution, t)
    prob = sol.problem(t) 
    sol.sol[t] = solve(prob, Tsit5()) 
end

function get_solution(sol::LSolution, x, t)
    in(t, keys(sol.sol)) && return sol.sol[t](x)
    new_solution!(sol, t)
    return sol.sol[t](x)
end

(sol::LSolution)(x,t) = get_solution(sol, x, t)

mutable struct TestFunction
    fn
    ts
    function TestFunction(ts, vals)
        f = new(nothing, ts)
        update_interp!(f, vals)        
        return f
    end
end

function update_interp!(fn::TestFunction, vals)
    fn.fn = linear_interpolation(fn.ts, vals, extrapolation_bc=Periodic())
end

function get_value(fn::TestFunction, t)
    return fn.fn(t)
end

(fn::TestFunction)(t) = get_value(fn, t)

function hazard(d::Gamma, x)
    return exp(logpdf(d, x) - logccdf(d, x))
end

function gammahaz(μ,cv,x) 
    iszero(μ) && return 0.0
    g = Gamma(1/cv, μ*cv)
    x > μ + 5*std(g) && return 1/(μ*cv)
    return hazard(g, x)
end

function solvef(λ, fn, xs, periods; kernel)
    ys = [kernel(x, t, λ) * fn(t-x) for x in xs, t in periods]
    problem = SampledIntegralProblem(ys, xs; dim = 1)
    return solve(problem, TrapezoidalRule())
end

function solvef1(λ, fn, xs, periods; kernel)
    ys = [diagm(kernel(x, t, λ)[1, :]) * fn(t-x) for x in xs, t in periods]
    problem = SampledIntegralProblem(ys, xs; dim = 1)
    sol = solve(problem, TrapezoidalRule())
    return TestFunction(periods, sol.u)
end


function update_f!(λ, fn, xs, periods; kernel)
    fu = solvef(λ, fn, xs, periods; kernel=kernel)
    update_interp!(fn, fu.u)
end

function lambda_integral(fn, xs, periods; kernel)
    integral(λ) = begin
        integrand(x, t) = exp(-λ*x) * kernel(x, t, λ) * fn(t-x)
        ys = [integrand(x, t) for x in xs, t in periods]
        problem = SampledIntegralProblem(ys, xs; dim = 1)
        soln = solve(problem, TrapezoidalRule())

        problem = SampledIntegralProblem(soln.u, periods; dim = 1)
        soln = solve(problem, TrapezoidalRule())
        return soln
    end
    return integral
end

function compute_lambda(fn, xs, periods; kernel)
    integral = lambda_integral(fn, xs, periods; kernel)
    zero = find_zero(λ -> 1 - sum(integral(λ).u), 1.0, Order0(), atol=1e-4, rtol=1e-4)
    return zero
end

function check_convergence(history, reltol)
    length(history) <= 10 && return false
    min_ = minimum(history[end-10:end])
    max_ = maximum(history[end-10:end])
    
    (max_ - min_)/min_ < reltol && return true
    return false
end

function iterate(xs, periods, maxiters; initfn, kernel, reltol=1e-3)  
    if isnothing(initfn)
        initf = ones(length(periods))
        initf = initf ./ (periods[end] - periods[1])
        fn = TestFunction(periods, collect.(zip(initf, zeros(length(periods)))))
    else
        fn = TestFunction(periods, initfn.(initfn.ts))
    end

    λ = 0.0
    λprev = Inf
    n = 1
    history = [] 
    converged = false
    while n <= maxiters && !converged
        telapse = @elapsed begin
            λ = compute_lambda(fn, xs, periods; kernel=kernel)
            push!(history, λ)
            update_f!(λ, fn, xs, periods; kernel=kernel)
            converged = check_convergence(history, reltol) 
            λprev = λ
        end
        println("λ estimate $λ... iteration took $(telapse) seconds")
        n += 1
    end

    fu1 = solvef1(λ, fn, xs, periods; kernel=kernel)
    return (λ, fn, fu1, history, converged)
end

# Note to self, first dimension rows.
mutable struct LSolutionInterp
    solution 
    xs
    ts
    
    function LSolutionInterp(fmatL, xs, ts, p)
        Lint(u, t) = diag(fmatL([u,], p, t))
        # Ls = xs rows, ts columns
        Ls = [Lint(u, t) for u in xs, t in ts]  
        n = length(xs) 
        m = length(ts) 
        sols = fill([1.0, 1.0], n, m)
          
        for i in 1:m
            for j in 1:n 
                xs_ = xs[1:j]
                length(xs_) < 2 && continue
                prob = SampledIntegralProblem(Ls[:, i][1:j], xs_)
                sols[j, i] = exp.(solve(prob, TrapezoidalRule()).u)
            end
        end
        itp = linear_interpolation((xs, ts), sols; extrapolation_bc=(Linear(), Periodic()) )
        new(itp, xs, ts)
    end
end

(L::LSolutionInterp)(x,t) = diagm(L.solution(x, t))

function run_opt(matM, matL, sys, param_map, xspan, period; initfn=nothing, iters=50, xintsteps=1000, tintsteps=1000, kwargs...)
    p = ModelingToolkit.MTKParameters(sys, param_map)
    fmatL, fmatL! = generate_custom_function(sys, matL; expression = Val(false))
    fmatM, fmatM! = generate_custom_function(sys, matM; expression = Val(false))

    xs = range(xspan[1], stop=xspan[2], length=xintsteps)
    periods = range(period[1], stop=period[2], length=tintsteps)
    
    lsol = LSolutionInterp(fmatL, xs, periods, p;)
    kernel(x, t, λ) = exp(-λ*x)*fmatM([x,], p, t)*lsol(x, t-x)
    return iterate(xs, periods, iters; kernel = kernel, initfn = initfn, kwargs...)
end

function main(matM, matL, sys; param_map, trange, iters, xintsteps, tintsteps, name, kwargs...)
    savepath = "data/$name/iters_$(iters)_intgrid_$(xintsteps)"
    mkpath(savepath)

    df_growth_rate = DataFrame(
        T1 = Float64[], 
        T2 = Float64[], 
        growth_rate = Float64[], 
        growth_rate_history = Vector{Vector{Float64}}(), 
        converged = Bool[], 
        file = String[])
    
    fn = nothing

    for T in trange 
        df_function = DataFrame(
            ts = Float64[], 
            f1 = Float64[], 
            f2 = Float64[], 
            f11 = Float64[], 
            f12 = Float64[])

        T1 = T
        T2 = T
        println("Performing iteration for T1 = $(T1), T2 = $(T2)...")
        param_map[pT] = [T1, T2]        

        xspan = (0.0, param_map[pd][1] + 2*(T1+T2))
        period = (0.0, (T1+T2))
        
        λ, fn, fn1, λhistory, converged = run_opt(
            matM, matL, sys,
            param_map, xspan, period; 
            initfn=fn, iters=iters, xintsteps=xintsteps, tintsteps=tintsteps, kwargs...)
        
        filename_fn = "$(savepath)/results_fn_T1_$(T1)_T2_$(T2).csv"
        push!(df_growth_rate, [T1, T2, λ, λhistory, converged, filename_fn])
                
        fx = fn.(fn.ts)
        fx1 = fn1.(fn1.ts)
        push!.(Ref(df_function), 
            collect(zip(collect(fn.ts), 
            getindex.(fx, 1), 
            getindex.(fx, 2), 
            getindex.(fx1, 1), 
            getindex.(fx1, 2))); promote=true)

        CSV.write(filename_fn, df_function)
    end
  
    filename_growth_rate = "$(savepath)/results_growth_rate.csv"
    CSV.write(filename_growth_rate, df_growth_rate)
end

function plot_lambda(;iters, xintsteps, name)
    savepath = "data/$(name)/iters_$(iters)_intgrid_$(xintsteps)"
    file = "/results_growth_rate.csv"

    res = DataFrame(CSV.File(savepath*file))
    fig = Figure()
    ax = Axis(fig[1,1]; xlabel="ln T", ylabel="Growth rate")
    
    scatterlines!(ax, log.(res[!, :T1]), res[!, :growth_rate])
    save("plots/dormancy/$(name)_growth_rate.pdf", fig)
end

function plot_f(;iters, xintsteps, name)
    savepath = "data/$(name)/iters_$(iters)_intgrid_$(xintsteps)"
    file = "/results_growth_rate.csv"
    res = DataFrame(CSV.File(savepath*file))
    
    n = length(eachrow(res))

    fig = Figure(size=(500, 300*n))
    axs = [Axis(fig[i,1]; title="T1 = $(f[:T1])", xlabel="T", ylabel="Density") for (i,f) in enumerate(eachrow(res))]
    
    for (i,r) in enumerate(eachrow(res))
        ffile = r[:file] 
        fres = DataFrame(CSV.File(ffile))

        lines!(axs[i], fres[!, :ts], fres[!, :f2]; color=colors[3], label="f2")
        lines!(axs[i], fres[!, :ts], fres[!, :f11]; color=colors[1], label="f11")
        lines!(axs[i], fres[!, :ts], fres[!, :f12]; color=colors[2], label="f12")
        axislegend(axs[i])
    end
    save("plots/dormancy/$(name)_fs.pdf", fig)
end
