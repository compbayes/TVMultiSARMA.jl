using TVMultiSARMA, Plots, LaTeXStrings, Measures, Random
using JLD2, Dates
using DynamicGlobalLocalShrinkage

experimentName = "usip"
mainFolder = joinpath(dirname(@__DIR__), experimentName)
resultsFolder = mainFolder*"/results"
dataFolder = mainFolder*"/data"
figFolder = mainFolder*"/figs"

Random.seed!(123) # Set seed for reproducibility

# Plot settings
colors = ["#6C8EBF", "#c0a34d", "#780000", "#007878", "#b5c6df","#eadaaa"]

gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=8,
    xtickfontsize=8, ytickfontsize=8, xguidefontsize=8, yguidefontsize=8,
    titlefontsize = 10, markerstrokecolor = :auto)

PACFmapping = "monahan"
θupdate = :ffbsx
nParticles = 100   # Number of PGAS particles
nInitFFBS = 500     # Number of FFBS iterations before PGAS
initVal = "fixed"   # Initial value for the state sampling. "prior" or "fixed"

# Model settings
season = [1,12]     # Seasonal periods
pFit = [1,2]        # Number of fitted lags in each AR polynomial
SV = true           # Fit stochastic volatility model for measurement errors, εₜ
nIter = 10000       # Number of MCMC iterations after burn-in
nBurn = 3000        # Number of burn-in MCMC iterations
offset = eps()      # offset log-volatility. A Float64 or "kowal"

modelName = experimentName*"SAR$(pFit[1])$(pFit[2])"

# Load data
data = load(dataFolder*"/usipdata_long.jld2")
x = data["x"]
dates = data["dates"]

activeLags = FindActiveLagsMultiSAR(pFit, season) # Non-zero lags in ϕ(L)Φ(L^s) poly

# Prior mean of σₑ from OLS fit with all lags.
ϕ̂, sₑ = SARMAasReg(x, pFit, season, imposeZeros = false)

μ₀, Σ₀ = NormalApproxUniformStationary(pFit)
priorSettings = (
    ϕ₀ = 0.5, κ₀ = 0.3, ν₀ = 3, ψ₀ = 1, m₀ = -15, σ₀ = 3, νₑ = 3, ψₑ = sₑ, μ₀ = μ₀, Σ₀ = Σ₀,
    ϕ̄₀ = 0.86, κ̄₀ = 0.11, ν̄₀ = 3, ψ̄₀ = 0.1, m̄₀ = log(sₑ^2), σ̄₀ = 3 # SV parameters
); 

modelSettings = (p = pFit, season = season, ztrans = PACFmapping, 
    fixσₙ = 1, α = 1/2, β = 1/2, intercept = false, SV = SV) 
algoSettings = (θupdate = θupdate, nIter = nIter, nBurn = nBurn, 
    nParticles = nParticles, nInitFFBS = nInitFFBS, initVal = initVal, 
    offsetMethod = eps()       
) 

# Run Gibbs
timing =  @elapsed θpost, Hpost, σₑpost, ϕpost, σ²ₙpost, μpost, ϕARpost = 
    GibbsLocalMultiSAR(x, modelSettings, priorSettings, algoSettings);
