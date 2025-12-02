
""" 
    PostSpecDensMultiSAR(ϕARpost, σₑpost, p, season; ωgrid = 0.01:0.01:π, thinFactor = 100) 

Compute posterior draws of the spectral density for a time-varying multi-season SAR model from posterior draws of the AR coefficients (`ϕARpost` a T×sum(p)×nIter matrix) at every `thinFactor` iteration.

""" 
function PostSpecDensMultiSAR(ϕARpost, σₑpost, p, season; ωgrid = 0.01:0.01:π, 
    thinFactor = 10, activeLags = nothing)

nIter  = size(ϕARpost,3)
T = size(ϕARpost, 1) # Including t=0
thinFactor = minimum(Int, [round(nIter/100), thinFactor])
println("using every $(thinFactor)th iteration to compute spectral density")
thinIter = thinFactor:thinFactor:nIter
specDensDraws = Array{Union{Float64,Missing}}(undef, length(ωgrid), T, length(thinIter))
if isnothing(activeLags)
    ϕ = Vector(undef, length(season))
    θ = repeat([[]], length(season)) # no MA coeff
else 
    ϕ = [zeros(sum(p .* season))] # models without multiplicative is like single period AR
    θ = [[]] # no MA coeff
    season = [1]
end
 
stationaryProp = zeros(T)
nIterThin = length(thinIter)
@showprogress for (c,i) in enumerate(thinIter)
       
    for t = 1:T # t = 1 is really t=0
          
        stationary = true
        if isnothing(activeLags)
            count = 0
            for l in 1:length(season)
                ϕ[l] = ϕARpost[t,(count + 1):(count + p[l]),i]
                stationary = stationary*check_stationarity(ϕ[l])[1]
                count = count + p[l]
            end
        else
            ϕ[1][activeLags] .= ϕARpost[t,:,i]
            stationary = check_stationarity(ϕ[1])[1]
        end

        if stationary
            specDensDraws[:,t,c] = SpecDensMultiSARMA(ωgrid, ϕ, θ, σₑpost[i]^2, season)
            stationaryProp[t] += 1/nIterThin
        else
            specDensDraws[:,t,c] .= missing
        end

    end
end

return specDensDraws, stationaryProp

end




""" 
    plotEvolStabilityRegion(ϕevol, θevol, iter, plotEvolLine = true) 

Plot the parameter evoluation in θ-space and ϕ-space (triangle in AR(2) case) up to iteration `iter`. `plotEvolLine = true` plots a line connecting the last 10 iterations.

""" 
function plotEvolStabilityRegion(ϕevol, θevol, iter, plotEvolLine = true, 
        titlerestricted = [], titleunrestricted = [])
    
    p = size(ϕevol,2)
    isempty(θevol) ? showθ = false : showθ = true
    
    plt1 = []
    plt2 = []

    if isempty(titlerestricted) titlerestricted = "Time = $(iter)" end
    if isempty(titleunrestricted) titleunrestricted = "Time = $(iter)" end

    if p == 2
        xs = [-2, 2, 0, -2] # Vertices' x-component
        ys = [-1, -1, 1, -1] # Vertices' y-component

        # Create a polygon for the shaded triangle
        poly = Shape(xs, ys)

        # Create the plot
        plt1 = plot(poly, seriestype = :shape, linecolor = nothing, fillalpha = 0.2, 
            fillcolor = :lightgray, label = nothing, title = titlerestricted)
        scatter!(ϕevol[1:iter,1], ϕevol[1:iter,2], xlim = (-2.5,2.5), 
                ylim = (-1.5,1.5), xlab = L"\phi_{t,1}", ylab = L"\phi_{t,2}", 
                label = nothing, markerstrokecolor = :blues, markersize = 2, 
                margin = 10mm, zcolor = (0:1/iter:1).^5, color = :blues, colorbar = false)
        if plotEvolLine 
            if iter>10
                plot!(ϕevol[(iter-10):iter,1], ϕevol[(iter-10):iter,2], 
                    label = nothing)
            else
                plot!(ϕevol[1:iter,1], ϕevol[1:iter,2], 
                    label = nothing)
            end
        end
        if showθ
            minmax = maximum(abs.(θevol[1:iter]))
            plt2 = scatter(θevol[1:iter,1], θevol[1:iter,2], 
                xlim = (-minmax, minmax), ylim = (-minmax, minmax), 
                title = titleunrestricted, 
                xlab = L"\theta_{t,1}", ylab = L"\theta_{t,2}", label = nothing, 
                markerstrokecolor = :blues, markersize = 2,
                zcolor = (0:1/iter:1).^5, color = :blues, colorbar = false)
        end
    end

    if p == 3 
        plt1 = scatter(ϕevol[1:iter,1], ϕevol[1:iter,2], ϕevol[1:iter,3], 
                xlim = (-2.5,2.5), ylim = (-1.5,1.5), zlim = (-1.5,1.5), 
                xlab = L"\phi_1", ylab = L"\phi_2", zlab = L"\phi_3",
                label = nothing, markerstrokecolor = :blues, markersize = 2, 
                margin = 10mm, zcolor = (0:1/iter:1).^5, color = :blues, colorbar = false)
        if plotEvolLine 
            if iter>10
                plot!(ϕevol[(iter-10):iter,1], ϕevol[(iter-10):iter,2], 
                        ϕevol[(iter-10):iter,3], label = nothing)
            else
                plot!(ϕevol[1:iter,1], ϕevol[1:iter,2], ϕevol[1:iter,3],
                    label = nothing)
            end
        end
        if showθ
            minmax = maximum(abs.(θevol[1:iter]))
            plt2 = scatter(θevol[1:iter,1], θevol[1:iter,2], θevol[1:iter,3], 
                xlim = (-minmax, minmax), ylim = (-minmax, minmax), 
                zlim = (-minmax, minmax),
                xlab = L"\theta_1", ylab = L"\theta_2", zlab = L"\theta_3",
                label = nothing, markerstrokecolor = :blues, markersize = 2, 
                margin = 10mm, zcolor = (0:1/iter:1).^5, color = :blues, colorbar = false)
        end
    end 

    return plt1, plt2

end # end function


function PlotPriorPostHyperparameters(ϕpost, σ²post, μpost, priorSettings, modelSettings)

    ϕ₀, κ₀, ν₀, ψ₀, μ₀, σ₀ = priorSettings
    p = modelSettings.p

    gr(xtickfontsize=6, ytickfontsize=6, titlefontsize = 12, legendfontsize = 8)

    pltϕ = plot()
    ϕgrid = range(quantile.(Truncated(Normal(ϕ₀, κ₀), -1, 1), [0.001,0.999])..., 
        length = 1000)
    pltϕ = plot!(ϕgrid, pdf.(Truncated(Normal(ϕ₀, κ₀), -1, 1),  ϕgrid), label = "prior",
        c = :black, title = L"\phi")
    for k = 1:p
        kdens = kde(ϕpost[k,:])
        plot!(pltϕ, kdens.x, kdens.density, label = L"\phi_{%$k}", color = colors[k], 
            lw = 2)
    end

    μgrid = range(quantile.(Normal(μ₀, σ₀), [0.001,0.999])..., length = 1000)
    pltμ = plot(μgrid, pdf.(Normal(μ₀, σ₀),  μgrid), c = :black, label = "prior", 
        title = L"\mu")
    for k = 1:p
        kdens = kde(μpost[k,:])
        plot!(pltμ, kdens.x, kdens.density, label = L"\mu_{%$k}", color = colors[k], lw = 2)
    end

    σgrid = range(0.001, sqrt(quantile(ScaledInverseChiSq(ν₀, ψ₀^2), 0.995)), length = 1000)
    pltσ = plot(σgrid, pdf.(ScaledInverseChiSq(ν₀, ψ₀^2),  σgrid.^2) .*(2*σgrid), 
        c = :black, label = "prior", title = L"\sigma_\eta", legend = :topright)
    for k = 1:p
        kdens = kde(sqrt.(σ²post[k,:]))
        plot!(pltσ, kdens.x, kdens.density, label = L"\sigma_{\eta,%$k}", color = colors[k], lw = 2)
    end

    return plot(pltϕ, pltμ, pltσ, layout = (2,2), size = (800,600), margin = 5mm)

   

end


""" 
    PlotMultiSAREvolution(x, ϕevol, θevol, p, season) 

Plot the time series and the evolution of the multi-seasonal AR parameters ϕ, θ over time.
""" 
function PlotMultiSAREvolution(x, ϕevol, θevol, p, season)

    nSeasons = length(season)
    # Plot data and true ϕ and θ evolutions
    ptimeseries = plot(x, lw = 1.5, color = :gray, xlabel = "time, t", ylabel = L"x_t")

    pltϕ = []
    for l in 1:nSeasons
        if season[l] == 1 # regular lag
            titleText = L"\phi"*" - regular lags"
        else
            titleText = L"\phi"*" - seasonal lags at "*L"s=%$(season[l])"
        end
        push!(pltϕ, plot(title = titleText, xlab = "time"))
        for j = 1:p[l] 
            plot!(ϕevol[l][:,j], label = L"\phi_{%$l%$(j)t}", c = colors[j])
        end
    end

    pltθ = []
    for l in 1:nSeasons
        if season[l] == 1 # regular lag
            titleText = L"\theta"*" - regular lags"
        else
            titleText = L"\theta"*" - seasonal lags at "*L"s=%$(season[l])"
        end
        push!(pltθ, plot(title = titleText, xlab = "time"))
        for j = 1:p[l] 
            plot!(θevol[l][:,j], label = L"\theta_{%$l%$(j)t}", c = colors[j])
        end
    end

    return ptimeseries, pltϕ, pltθ

end


""" 
    PlotCompanionEigen(ϕevol) 

Computes eigenvalues of the companion matrix for an evolving AR process

""" 
function PlotCompanionEigenMultiSAR(ϕevol, season; jitter = 0.0)
    
    nSeasons = length(season)
    p = [size(ϕevol[l], 2) for l in 1:length(ϕevol)]
    T = size(ϕevol[1], 1)
    colorgradients = []
    push!(colorgradients,
        [cgrad(:blues, scale= :linear)[z] for z ∈ (1:T)/T],
        [cgrad(colors[[6,2]], scale= :linear)[z] for z ∈ (1:T)/T],
        [cgrad(colors[[3,12]], scale= :linear)[z] for z ∈ (1:T)/T]
    )

    thetaGrid = LinRange(0, 2π, 100)
    x = cos.(thetaGrid)
    y = sin.(thetaGrid)
    plts = []
    for l in 1:nSeasons
        if season[l] == 1
            titleText = "Eigen ϕ at regular lags"
        else
            titleText = "Eigen ϕ at season $(season[l])"
        end
        push!(plts, plot(x, y, aspect_ratio=:equal, color = :gray, legend=false, 
            xlabel = "real", ylabel = "imaginary", lw = 1, title = titleText))
        for t = 1:T 
            eig = check_stationarity(ϕevol[l][t,:])[2]
            for j = 1:p[l]
                scatter!([real(eig[j])], [imag(eig[j]) + (t/T)*jitter ], ms = 2.5, 
                    color = colorgradients[j][t])
            end
        end
    end

    return plts
end