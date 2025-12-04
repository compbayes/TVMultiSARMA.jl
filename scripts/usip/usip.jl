# # Time-varying seasonal AR model for monthly US industrial production data

# In this example we analyze the TVSAR(p,P) model in Fagerberg et al. (2025) with p regular lags and P seasonal lags at seasonality s = 12 and stochastic volatility for the measurement errors:
#
# ```math
# \begin{align*}
#   \phi_t(L)\Phi_t(L^s)y_t &= \epsilon_t , \quad \epsilon_t \sim N(0,\sigma_\varepsilon_t^2) \\
#   \log \sigma_{\varepsilon_t}^2 &= \log \sigma_{\varepsilon_{t-1}}^2 + \xi_t, \quad \sim N(0,\sigma_\xi_t^2)
# \end{align*}
# ```
# where $\phi_t(L) = 1 - \phi_{1t} L - \phi_{2t} L^2 - \ldots - \phi_{pt} L^p$ is the regular AR polynomial and $\Phi_t(L^s) = 1 - \Phi_{1t} L^s - \Phi_2 L^{2s} - \ldots - \Phi_{Pt} L^{Ps}$ is the seasonal AR polynomial.
# The AR parameters follow independent dynamic shrinkage process priors.

# ### Load some packages, set plotting backend and random seed:
using TVMultiSARMA, Plots, LaTeXStrings, Measures, Random
using JLD2, Dates, DSP, Downloads
using TimeSeriesUtils
using Utils: quantile_multidim, optimalPlotLayout
using Utils: mvcolors as colors

gr(legend = nothing, grid = false, color = colors[1], lw = 1, legendfontsize=12,
    xtickfontsize=8, ytickfontsize=8, xguidefontsize=10, yguidefontsize=10,
    titlefontsize = 12)

Random.seed!(123) # Set seed for reproducibility

# ### Load and setup the data
data = load(Downloads.download("https://github.com/mattiasvillani/TVMultiSARMA.jl/raw/main/scripts/usip/data/usipdata_long.jld2"))
x = data["x"]
dates = data["dates"]
dateticks = Date(Year(1920)):Year(20):Date(Year(2024))
plot(dates, x, xlabel = "time", ylabel = "USIP", xticks = (dateticks, year.(dateticks)),
    title = "US industrial production - level detrended")

# ### Model, algorithm and prior settings

# #### Model settings
season = [1,12]     # Seasonal periods
pFit = [1,2]        # Number of fitted lags in each AR polynomial
SV = true           # Fit stochastic volatility model for measurement errors, εₜ
nIter = 1000       # Number of MCMC iterations after burn-in
nBurn = 300        # Number of burn-in MCMC iterations
offsetMethod = eps(); # offset log-volatility. A Float64 or "kowal"
pacf_map = "monahan" # Map from θ to pacf's: "monahan", "tanh", "hardtanh", "linear".

modelSettings = (p = pFit, season = season, pacf_map = pacf_map, 
    fixσₙ = 1, α = 1/2, β = 1/2, intercept = false, SV = SV)

# #### Algorithm settings
θupdate = :ffbsx     # Sampling of states: :pgas, :ffbs, :ffbsx, :ffbs_unscented
nParticles = 100     # Number of :pgas particles
nInitFFBS = 500      # Number of :ffbsx iterations before :pgas
initVal = "fixed";   # Initial value for the sampling of the state. "prior" or "fixed"

algoSettings = (θupdate = θupdate, nIter = nIter, nBurn = nBurn, 
    nParticles = nParticles, nInitFFBS = nInitFFBS, initVal = initVal, 
    offsetMethod = offsetMethod       
) 

# #### Prior settings
## Prior mean of measurement stdev σₑ from OLS fit with all lags.
ϕ̂, sₑ = SARMAasReg(x, pFit, season, imposeZeros = false)

## Prior for state at t=0 from Normal approximation to uniform dist over stability region
μ₀, Σ₀ = NormalApproxUniformStationary(pFit)

priorSettings = (
    ϕ₀ = 0.5, κ₀ = 0.3, ν₀ = 3, ψ₀ = 1, m₀ = -15, σ₀ = 3,           # DSP hyperparameters
    νₑ = 3, ψₑ = sₑ, μ₀ = μ₀, Σ₀ = Σ₀,                              # err var and state t=0
    ϕ̄₀ = 0.86, κ̄₀ = 0.11, ν̄₀ = 3, ψ̄₀ = 0.1, m̄₀ = log(sₑ^2), σ̄₀ = 3  # SV parameters
); 

# ### Run Gibbs sampling
θpost, Hpost, σₑpost, ϕpost, σ²ₙpost, μpost, ϕARpost = GibbsLocalMultiSAR(x, modelSettings,
    priorSettings, algoSettings);

# ### Plotting the posterior of the parameter evolution

## Plot posterior parameter evolution for AR coefficients
ylimits = [(0,1), (-0.5,1), (-0.5,1)] # ylims for each subplot
T_orig = length(x)
T_eff = size(ϕARpost, 1)
Tgrid = (T_orig-T_eff + 1):T_orig
plts = [] 
for l = 1:length(pFit)
    for j in 1:pFit[l]
        plt = plot(xticks = (dateticks, year.(dateticks)), 
            ylims = ylimits[j+(l-1)*pFit[1]])
        ϕAR_tmp = ConvertWideMat2Vec(ϕARpost, pFit, season)
        quants = quantile_multidim(ϕAR_tmp[l], [0.025, 0.5, 0.975]; dims = 3) 
        plot!(plt, dates[Tgrid], quants[:,j,1], color = colors[1])
        plot!(plt, dates[Tgrid], quants[:,j,2], color = colors[3])
        plot!(plt, dates[Tgrid], quants[:,j,3], color = colors[1])
        title!((l == 1) ? L"\phi_{%$(j)t}" : L"\Phi_{%$(j)t}")
        push!(plts, plt)
    end 
end 

## Plot posterior parameter evolution for measurement standard deviation σₑ
if SV
    plt = plot(title = L"\sigma_{t}", legend = nothing, size = (600,300))
    quants = quantile_multidim(σₑpost, [0.025, 0.5, 0.975]; dims = 2) 
    plot!(plt, dates[Tgrid[2:end]] , quants[:,1], color = colors[1],
        xticks = (dateticks, year.(dateticks)), label = L"95\%"*" C.I.")
    plot!(plt, dates[Tgrid[2:end]], quants[:,2], color = colors[3], label  = "median")
    plot!(plt, dates[Tgrid[2:end]], quants[:,3], color = colors[1], label = nothing)
    push!(plts, plt)
end

plot(plts..., layout = optimalPlotLayout(length(plts)), size = (800,400), 
    margin = 2mm)

