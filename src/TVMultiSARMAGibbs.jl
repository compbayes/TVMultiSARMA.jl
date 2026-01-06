# Function for converting log volatilities to Cov matrices used in the filters
LogVol2Covs(H) = PDMat.([diagm(exp.(H[t,:])) for t in 1:size(H,1)])

"""
    GibbsLocalMultiSAR(x, modelSettings, priorSettings, algoSettings)

Gibbs sampling from the posterior of the locally stable time-varying multi-seasonal AR(p) model with dynamic shrinkage process prior. Optionally, with stochastic volatility.

""" 
function GibbsLocalMultiSAR(x, modelSettings, priorSettings, algoSettings, 
    initialValues = nothing)

    # Unpack
    ϕ₀, κ₀, ν₀, ψ₀, m₀, σ₀, νₑ, ψₑ, μ₀, Σ₀, ϕ̄₀, κ̄₀, ν̄₀, ψ̄₀, m̄₀, σ̄₀ = priorSettings
    p, season, pacf_map, fixσₙ, α, β, includeIntercept, SV = modelSettings
    θupdate, nIter, nBurn, nParticles, nInitFFBS, initVal, offsetMethod = algoSettings 
    updateσₙ = isnothing(fixσₙ)
    isa(offsetMethod, Function) ? offset = eps() : offset =  offsetMethod # init offset DSP
 
    if any(season .> 1)  && (θupdate == :ffbs) 
        error("ffbs requires a linear Gaussian model and cannot be used with seasonal AR")
    end
    
    # Write AR(p) as regression and loose p first observations, so T -> T- p_max
    p_max = sum(p .* season)
    nLags = sum(p) # Total number of lags in all polynomials
    y, Z, T = SetupARReg(x, p_max)

    # Find the active lags and select the corresponding columns of Z
    activeLags = FindActiveLagsMultiSAR(p, season)
    Z = Z[:,activeLags] # Select the lags with non-zero coeff when poly are multiplied

    ## Set up prior, transition and observation models
    prior = (nLags == 1) ? Normal(μ₀[1], sqrt(Σ₀[1])) : MvNormal(μ₀, Σ₀)

    transition(param, state, t) = (nLags == 1) ? Normal(state, sqrt(param.Σᵥ[t][1])) : 
        MvNormal(state, param.Σᵥ[t])

    observation(param, state, t) = Normal(param.Z[t,:]⋅MultiSARtoReg(state, p, season,  
        activeLags; pacf_map = pacf_map), sqrt(param.Σₑ[t][1]))
   

    # Measurement function and its derivative for FFBSx
    C(θ, z) = z ⋅ MultiSARtoReg(θ, p, season, activeLags; pacf_map = pacf_map)
    ∂C(θ, z) = ForwardDiff.gradient(θ -> C(θ, z), θ)' # gradient of measurement function
    
    # Approximate the log χ²₁ distribution with a mixture of normals
    nMixComp = 10
    mixture = SetUpLogChi2Mixture(nMixComp) 
    P = zeros(T, nMixComp) # Posterior probabilities for mixture components, updated later

    # Storage
    θpost = zeros(T+1, nLags, nIter) # The initial θ₀ goes first here.
    Hpost = zeros(T, nLags, nIter)
    σₑpost = zeros(T, nIter)
    ϕpost = zeros(nLags, nIter)      # This stores κ in the model. 
    μpost = zeros(nLags, nIter)
    σ²ₙpost = zeros(nLags, nIter) 
    ϕARpost = zeros(T+1, nLags, nIter)
    if SV
        ϕ̄post = zeros(nIter)
        μ̄post = zeros(nIter)
        σ̄²ₙpost = zeros(nIter)
    else
        ϕ̄post = nothing
        μ̄post = nothing
        σ̄²ₙpost = nothing
    end
    θparticles = zeros(nParticles, sum(p), T+1) # Initialize PGAS particle container.

    # Initial values
    if (nInitFFBS>0) && (θupdate == :pgas)
        println("Getting initial values from FFBSx")
        A = PDMat(I(nLags))
        B = zeros(nLags)
        Cargs = [Z[t,:] for t in 1:T]
        U = zeros(T,1)
        algoSettingsInit = (θupdate = :ffbsx, nIter = nInitFFBS, 
            nBurn = round(Int, 0.1*nInitFFBS), nParticles = nParticles, nInitFFBS = 0, initVal = initVal, offsetMethod = offsetMethod)
        θpostInit, HpostInit, σₑpostInit, ϕpostInit, σ²ₙpostInit, μpostInit, ϕARpostInit =  
            GibbsLocalMultiSAR(x, modelSettings, priorSettings, algoSettingsInit);
        θ = median(θpostInit, dims = 3)
        H = median(HpostInit, dims = 3)
        σₑ = median(σₑpostInit, dims = 2)
        h̄ = log.(σₑ.^2)
        ϕ = median(ϕpostInit, dims = 2)
        μ = median(μpostInit, dims = 2)
        ξ = ones(T, nLags)    # Polya-gamma variables
        ϕ̄ = ϕ̄₀
        μ̄ = m̄₀
        σ̄²ₙ = ψ̄₀^2
        if updateσₙ
            σ²ₙ = median(σ²ₙpostInit, dims = 2)
        else
            σ²ₙ = (fixσₙ^2)*ones(nLags)
        end
    else
        θ = zeros(T+1, nLags) # The initial θ₀ goes first here.               
        ξ = ones(T, nLags)    # Polya-gamma variables
        σₑ = ψₑ*ones(T)
        if initVal == "prior"
            ϕ = rand(Truncated(Normal(ϕ₀, κ₀), -1, 1))*ones(nLags)
            μ = rand(Normal(m₀, σ₀))*ones(nLags)
            ϕ̄ = rand(Truncated(Normal(ϕ̄₀, κ̄₀), -1, 1))
            μ̄ = rand(Normal(m̄₀, σ̄₀))
            σ̄²ₙ = rand(ScaledInverseChiSq(ν̄₀, ψ̄₀^2))
        else # fixed initial values
            ϕ = ϕ₀*ones(nLags)
            μ = m₀*ones(nLags)
            ϕ̄ = ϕ̄₀
            μ̄ = m̄₀
            σ̄²ₙ = ψ̄₀^2
        end
        if isnothing(fixσₙ) # estimate
            σ²ₙ = (ψ₀^2)*ones(nLags)
        else # Fix, don't estimate
            σ²ₙ = (fixσₙ^2)*ones(nLags) 
        end
        h̄ = log.(σₑ.^2)
        H = repeat(μ', T, 1)
         
    end
    if !isnothing(initialValues) # Overwrite with user specified initial values
        θ = initialValues.θ
        H = initialValues.H
        μ = initialValues.μ
        ϕ = initialValues.ϕ
    end
    ξ̄ = ones(T) # Trick. No Polya-Gamma for SV. Allows us to use the same update function. 
    H̃ = H .- μ'

    ϕAR = zeros(T+1, nLags)
    S = zeros(Int, T, nLags) # Mixture allocation for χ²₁ - hₜ model
    Dᵩ = BandedMatrix(-1 => repeat([-ϕ[1]], T-1), 0 => Ones(T)) # Init D matrix for h_t
    D̄ = BandedMatrix(-1 => repeat([-ϕ̄], T-1), 0 => Ones(T)) # Init D matrix SV

    Σᵥ = LogVol2Covs(H)
    Σₑ = LogVol2Covs(h̄) # Slightly misleading notation. Elements in Σₑ are still a scalars

    # Set up model for θupdate
    if θupdate == :pgas
        A = I(nLags)
        param = (Σᵥ = Σᵥ, Z = Z, Σₑ = Σₑ) 
        scalingFactor = 1
        if nInitFFBS > 0
            μ_prop = median(θpostInit[1,:,:], dims = 2)[:]
            Σ_prop = PDMat(cov(θpostInit[1,:,:], dims = 2))
            if nLags == 1
                initialization = Normal(μ_prop[1], sqrt(scalingFactor)*sqrt(Σ_prop[1,1]))
            else
                initialization = MvNormal(μ_prop, scalingFactor*Σ_prop)
            end
            θ = median(θpostInit, dims = 3) # Initial reference particle for pgas
        else
            initialization = prior
            θ = PGASsimulate!(θparticles, y, nLags, nParticles, param, 
                prior, transition, observation, initialization, systematic; 
                sample_t0 = true) # Initial reference particle for pgas from filter
        end        
    elseif θupdate == :ffbs
        A = I(nLags)
        B = zeros(nLags)
        C = zeros(1,nLags,T)
        for t = 1:T
            C[1,:,t] = Z[t,:]'
        end 
        U = zeros(T,1)
    elseif θupdate == :ffbsx || θupdate == :ffbs_unscented
        A = PDMat(I(nLags)) #FIXME: Do we need this PDMat?
        B = zeros(nLags)
        Cargs = [Z[t,:] for t in 1:T]
        U = zeros(T,1)
    end    

    @showprogress for i = 1:(nBurn+nIter)

        Σᵥ = LogVol2Covs(H) 
        Σₑ = LogVol2Covs(h̄) 

        # Update θ₀, θ₁, θ₂, ..., θ_T 
        if θupdate == :pgas
            param = (Σᵥ = Σᵥ, Z = Z, Σₑ = Σₑ)   
            θ = PGASsimulate!(θparticles, y, nLags, nParticles, param, 
                prior, transition, observation, initialization, systematic, θ; 
                sample_t0 = true) 
        elseif θupdate == :ffbs
            θ = FFBS(U, y, A, B, C, Σₑ, Σᵥ, μ₀, Σ₀)
        elseif θupdate == :ffbsx 
            θ = FFBSx(U, y, A, B, C, ∂C, Cargs, Σₑ, Σᵥ, μ₀, Σ₀)
        elseif θupdate == :ffbs_unscented
            θ = FFBS_unscented(U, y, A, B, C, Cargs, Σₑ, Σᵥ, μ₀, Σ₀, 
                α = 1, β = 0, κ = 1)
        end
        
        # Truncate
        if pacf_map == "monahan" # TODO: maybe remove? 
            clamp!(θ, -10.0, 10.0) 
        elseif pacf_map == "sigmoid"
            clamp!(θ, -6.0, 6.0) 
        end
            
        # Compute AR coefficients 
        count = 0
        for l in 1:length(season)
            ϕAR[:, (count + 1):(count + p[l])] = mapslices(
                θ -> arma_reparam(θ; pacf_map = pacf_map)[1], 
                θ[:,(count + 1):(count + p[l])], dims = 2
            )
            count = count + p[l]
        end
 
        # Update noise variance
        offsetSV = eps()
        h̄, ϕ̄, μ̄, σ̄²ₙ = UpdateErrorVolatility(y, Z, θ, h̄, ξ̄, ϕ̄, μ̄, σ̄²ₙ, 
            ϕ̄₀, κ̄₀, m̄₀, σ̄₀, ν̄₀, ψ̄₀, νₑ, ψₑ, mixture, p, season, activeLags, pacf_map, SV, offsetSV)
        if !SV
            h̄ = h̄*ones(T) # homoscedastic errors
        end
        σₑ = exp.(h̄/2)

        # Update the log-volatility evolution H and DSP static parameters
        ν = diff(θ, dims = 1) # Param evolution matrix, vₜ,ₖ = θₜ,ₖ - θₜ₋₁,ₖ for t = 1,...T
        setOffset!(offset, ν, offsetMethod)
        update_dsp!(ν, S, P, H, H̃, ξ, ϕ, μ, σ²ₙ, priorSettings, mixture, Dᵩ, 
            offset, α, β, updateσₙ)
     
        # store
        if i>nBurn
            θpost[:,:,i-nBurn] = θ
            Hpost[:,:,i-nBurn] = H
            σₑpost[:,i-nBurn] .= σₑ 
            ϕpost[:,i-nBurn] = ϕ
            σ²ₙpost[:,i-nBurn] = σ²ₙ
            μpost[:,i-nBurn] = μ
            ϕARpost[:,:,i-nBurn] = ϕAR
            if SV
                ϕ̄post[i-nBurn] = ϕ̄ 
                μ̄post[i-nBurn] = μ̄
                σ̄²ₙpost[i-nBurn] = σ̄²ₙ
            end
        end

    end

    return θpost, Hpost, σₑpost, ϕpost, σ²ₙpost, μpost, ϕARpost, ϕ̄post, μ̄post, σ̄²ₙpost

end 


# Draw the stochastic volatility - log(σₑ²)
function UpdateErrorVolatility(y, Z, θ, h̄, ξ̄, ϕ̄, μ̄, σ̄²ₙ, 
        ϕ̄₀, κ̄₀, m̄₀, σ̄₀, ν̄₀, ψ̄₀, νₑ, ψₑ, mix, p, season, activeLags, pacf_map, SV = true, offsetSV = eps())
    ϕ̃ = mapslices(θ -> MultiSARtoReg(θ, p, season, activeLags; pacf_map = pacf_map), 
        θ, dims = 2)
    residuals = y - sum(Z .* ϕ̃[2:end,:], dims = 2) # Residuals for the AR(p) model
    T = length(residuals)
    if !SV
        return log(rand(ScaledInverseChiSq(νₑ + T, 
            (νₑ*ψₑ^2 + sum(residuals.^2))/(νₑ + T)))), nothing, nothing, nothing
    end
    # Stochatic volatility
    ȳ = log.(residuals.^2 .+ offsetSV)[:]
    s̄ = UpdateMixAlloc(ȳ, h̄, mix.dist)[:,1] # mixture allocation for log χ²₁
    D̄ = BandedMatrix(-1 => repeat([-ϕ̄], T-1), 0 => Ones(T)) # Init D matrix SV
    h̄ = Update_h(ȳ, mix.m[s̄], mix.v[s̄], D̄, ξ̄, ϕ̄, σ̄²ₙ, μ̄)
    ϕ̄ = Updateϕ(h̄, ξ̄, μ̄, σ̄²ₙ, ϕ̄₀, κ̄₀)
    σ̄²ₙ = Updateσ²ₙ(h̄, ξ̄, ϕ̄, μ̄, ν̄₀, ψ̄₀)
    μ̄ = Updateμ(h̄, ξ̄, ϕ̄, σ̄²ₙ, m̄₀, σ̄₀)
    
    
    return h̄, ϕ̄, μ̄, σ̄²ₙ

end