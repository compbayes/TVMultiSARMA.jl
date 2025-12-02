using TVMultiSARMA, Plots, LaTeXStrings, Measures, Random
using JLD2, Dates, DSP
using DynamicGlobalLocalShrinkage
using TimeSeriesUtils
using Utils: quantile_multidim
using Utils: mvcolors as colors

gr(legend = nothing, grid = false, color = colors[1], lw = 1, legendfontsize=12,
    xtickfontsize=8, ytickfontsize=8, xguidefontsize=10, yguidefontsize=10,
    titlefontsize = 12)

Random.seed!(123) # Set seed for reproducibility

# Algorithm settings
pacf_map = "monahan" # Map from θ to pacf's: "monahan", "tanh", "hardtanh", "linear".
θupdate = :ffbsx     # Sampling of states: :pgas, :ffbs, :ffbsx, :ffbs_unscented
nParticles = 100     # Number of :pgas particles
nInitFFBS = 500      # Number of :ffbsx iterations before :pgas
initVal = "fixed"    # Initial value for the sampling of the state. "prior" or "fixed"

# Model settings
season = [1,12]     # Seasonal periods
pFit = [1,2]        # Number of fitted lags in each AR polynomial
SV = true           # Fit stochastic volatility model for measurement errors, εₜ
nIter = 10000       # Number of MCMC iterations after burn-in
nBurn = 3000        # Number of burn-in MCMC iterations
offsetMethod = eps() # offset log-volatility. A Float64 or "kowal"

# Load and setup the data
data = load(dataFolder*"/usipdata_long.jld2")
x = data["x"]
dates = data["dates"]
dateticks = Date(Year(1920)):Year(20):Date(Year(2024))
plot(dates, x, xlabel = "time", ylabel = "USIP", xticks = (dateticks, year.(dateticks)),
    title = "US industrial production - level detrended")

activeLags = FindActiveLagsMultiSAR(pFit, season) # Non-zero lags in ϕ(L)Φ(L^s) polynomial

# Prior mean of σₑ from OLS fit with all lags.
ϕ̂, sₑ = SARMAasReg(x, pFit, season, imposeZeros = false)

# Prior for state at t=0 from Normal approximation to uniform dist over stability region
μ₀, Σ₀ = NormalApproxUniformStationary(pFit)
priorSettings = (
    ϕ₀ = 0.5, κ₀ = 0.3, ν₀ = 3, ψ₀ = 1, m₀ = -15, σ₀ = 3,           # DSP hyperparameters
    νₑ = 3, ψₑ = sₑ, μ₀ = μ₀, Σ₀ = Σ₀,                              # err var and state t=0
    ϕ̄₀ = 0.86, κ̄₀ = 0.11, ν̄₀ = 3, ψ̄₀ = 0.1, m̄₀ = log(sₑ^2), σ̄₀ = 3  # SV parameters
); 

modelSettings = (p = pFit, season = season, pacf_map = pacf_map, 
    fixσₙ = 1, α = 1/2, β = 1/2, intercept = false, SV = SV) 
algoSettings = (θupdate = θupdate, nIter = nIter, nBurn = nBurn, 
    nParticles = nParticles, nInitFFBS = nInitFFBS, initVal = initVal, 
    offsetMethod = offsetMethod       
) 

# Run Gibbs
θpost, Hpost, σₑpost, ϕpost, σ²ₙpost, μpost, ϕARpost = GibbsLocalMultiSAR(x, modelSettings,
    priorSettings, algoSettings);


# Plot posterior of parameter evolution for AR parameters 
ylimits = [(0,1), (-0.5,1), (-0.5,1)] # ylims for each subplot
plts = Array{Any}(undef, sum(pFit) + SV)
titles = []
for l in 1:length(pFit)
    for i in 1:pFit[l]
        if l == 1
            push!(titles, L"\phi_{%$(i)t}")
        else
            push!(titles, L"\Phi_{%$(i)t}")
        end
    end
end
T_orig = length(x)
T_eff = size(ϕARpost, 1)
Tgrid = (T_orig-T_eff + 1):T_orig
pcount = 0 
for l = 1:length(pFit)
    for j in 1:pFit[l]
        pcount = pcount + 1
        plts[pcount] = plot(xticks = (dateticks, year.(dateticks)), 
            ylims = ylimits[pcount])
        ϕAR_tmp = ConvertWideMat2Vec(ϕARpost, pFit, season)
        quants = quantile_multidim(ϕAR_tmp[l], [0.025, 0.5, 0.975]; dims = 3) 
        plot!(plts[pcount], dates[Tgrid], quants[:,j,1], color = colors[1])
        plot!(plts[pcount], dates[Tgrid], quants[:,j,2], color = colors[3])
        plot!(plts[pcount], dates[Tgrid], quants[:,j,3], color = colors[1])
        title!(titles[pcount])
    end 
end 

# Plot posterior parameter evolution for measurement standard deviation σₑ
if SV
    pcount = pcount + 1
    push!(titles, L"\sigma_{t}")
    plts[pcount] = plot(title = titles[end], legend = nothing, size = (600,300))
    quants = quantile_multidim(σₑpost, [0.025, 0.5, 0.975]; dims = 2) 
    plot!(dates[Tgrid[2:end]] , quants[:,1], color = colors[1],
        xticks = (dateticks,datetickslabels), label = L"95\%"*" C.I.")
    plot!(dates[Tgrid[2:end]], quants[:,2], color = colors[3], label  = "median")
    plot!(dates[Tgrid[2:end]], quants[:,3], color = colors[1], label = nothing)
end

plot(plts..., layout = optimalPlotLayout(length(plts)), size = (800,400), 
    margin = 2mm)


# Empirical time-varying spectral density estimate
function tvPeriodogram(y, N, S, nFreq = nothing, taper = nothing; 
        taperArgs...)
    if isnothing(taper) 
      win_normalized = nothing
    else
      win = taper(Int64(N); taperArgs...);
      win_normalized = win/sqrt(mean(win.^2))
    end
    if isnothing(nFreq) 
      numFrequencies = nextfastfft(N)
    else
      numFrequencies = nextfastfft(2*nFreq)
    end
    noverlap = N - S
    specG = spectrogram(y, N, noverlap ; onesided=true,  nfft=numFrequencies, fs=1, 
        window = win_normalized);
  
    return specG.power ./ 4π , 2π*specG.freq
end

# Posterior of log-spectrogram from SAR model
ωgrid = 0.01:0.01:π
specDensDraws = PostSpecDensMultiSAR(ϕARpost, σₑpost, pFit, season; ωgrid  = ωgrid, 
    thinFactor = 10);
quantilesLogSpecDens = quantile_multidim(log.(specDensDraws), [0.025, 0.5, 0.975], dims = 3)
quantilesLogSpecDens = quantilesLogSpecDens[:,2:end,:]
lowLogSpecDens = quantilesLogSpecDens[:,:,1]'
medianLogSpecDens = quantilesLogSpecDens[:,:,2]'
highLogSpecDens = quantilesLogSpecDens[:,:,3]'

# Non-parametric estimate of time-varying spectral density
N = 120; # number of observations in each segment.
S = 36; # Number of steps the time window moves forward. Number of overlapping obs. is N-S.
M = (T_orig-N)÷S + 1 # This is the total number of segments we will use.
nFreq = length(ωgrid) # No of frequencies we want to use

tvI, freqs = tvPeriodogram(x, N, S, nFreq, hanning);
nFreq = size(tvI)[1]
j = 1:size(tvI)[1];
times_blocks = round.(Int,collect(range(N/2, T_orig-(N/2), length=size(tvI)[2])))
dates_blocks = dates[times_blocks]
ωgridSub = (π .*j) ./ size(tvI)[1];

extremelogtvI = extrema(log.(tvI))
extremeModel = extrema(medianLogSpecDens)
cLimits = (minimum([extremelogtvI[1],extremeModel[1]]), 
    maximum([extremelogtvI[2],extremeModel[2]]))

# Plot spectrogram with segment highlighted and points for middle of segment
colorgradients = :viridis
plt_heatmap_tvI = heatmap(dates_blocks, ωgridSub, log.(tvI), c = colorgradients, 
    legend = :none, title="log spectrogram", xlab = "time", ylab = "Frequency", 
    zlab = L"\log f(\omega)", xticks = (dateticks,datetickslabels), clims = cLimits,
    yticks = ([0, π/4,  π/2, 3π/4,  π], [L"0" L"\pi/4" L"\pi/2" L"3\pi/4" L"\pi"]), 
    xlims = (dates[Tgrid[2:end]][1], dates[Tgrid[2:end]][end]), colorbar = false)

plt_heatmap_SAR = heatmap(dates[Tgrid[2:end]], ωgrid, medianLogSpecDens', 
    c = colorgradients, clims = cLimits,
    xlab = "time", ylab = "Frequency", title = "SAR($(pFit[1]),$(pFit[2]))", 
        colorbar = false,
    yticks = ([0, π/4,  π/2, 3π/4,  π], [L"0" L"\pi/4" L"\pi/2" L"3\pi/4" L"\pi"]), 
        xticks = (dateticks,datetickslabels))

gr(legend = nothing, grid = false, color = colors[1], lw = 1, legendfontsize=14,
    xtickfontsize=8, ytickfontsize=8, xguidefontsize=10, yguidefontsize=10,
    titlefontsize = 10)
plot(plt_heatmap_tvI, plt_heatmap_SAR, layout = (1,2), size = (800,300), margin = 3mm)