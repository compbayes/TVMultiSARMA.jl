
""" 
    y, Z, T = SetupARReg(x, p) 

Sets up the lagged matrix Z and the response vector y for an AR(p) process. 

The AR(p) process x_t = ϕ₁*x_{t-1} + ... + ϕ_p*x_{t-p} + ν_t is transformed into the regression model:

y = Z*ϕ + ν

where y = [x_{p+1}, x_{p+2}, ..., x_T],  the t:th row of Z is 
Z[t,:] = [x_{t-1}, x_{t-2}, ..., x_{t-p}] and ϕ = [ϕ₁, ϕ₂, ..., ϕ_p].

Note: we loose the first p observations in the transformation and the return T is adjusted accordingly to T-p.

# Examples
```julia-repl
julia> x = simARMA([0.7,-0.2], [0.0], 0.0, 1.0, 100)
julia> y, Z, T = SetupARReg(x, p)
julia> inv(Z'*Z)*Z'y
```
""" 
function SetupARReg(x, p)
    T = length(x)

    # Set up lags
    y = x[p+1:end]
    Z = zeros(T, p)
    for i = 1:p
        Z[:,i] = [zeros(i); x[1:(end-i)]]
    end
    Z = Z[(p+1):end,:]
    T = T - p # Redefining time here!()

    return y, Z, T
end

""" 
    FindActiveLagsMultiSAR(p, s)

Find the active lags in a general multi-seasonal SAR(p,s) process with seasons in the vector `s`. The number of lags for each seasonal polynomial is given in the vector `p`.

""" 
function FindActiveLagsMultiSAR(p::AbstractVector, s::AbstractVector)
    ARpolynomials = Array{Polynomial}(undef, length(p))
    for (j, pⱼ) ∈ enumerate(p)
        ARpolymat = [zeros(s[j]-1, pⱼ);-ones(1,pⱼ)]
        ARpolynomials[j] = Polynomial([1;ARpolymat[:]], :z)
    end
    return (findall(coeffs(prod(ARpolynomials)) .!= 0) .- 1)[2:end]
end


""" 
    μ₀, Σ₀ = NormalApproxUniformStationary(p) 

Returns mean vector (μ₀) and covariance matrix (Σ₀) in the θ ~ Normal approximation to the uniform prior for the unrestricted θₖ in the multi-seasonal AR with `p` regular lags and seasonal lags.

For example, p = [2,2,1] is SAR model with two regular lags, two lags at the first seasonal period (e.g. daily cycle) and a single lag on the second seasonal period (e.g. yearly cycle).
The approximation minimizes the Hellinger distance to the t and skew-t priors.

# Examples
```julia-repl
julia> μ₀, Σ₀ = NormalApproxUniformStationary([2, 1])
([0.0, -0.529625, 0.0], [1.0857014809 0.0 0.0; 0.0 0.7363356099999999 0.0; 0.0 0.0 1.0857014809])
```
""" 
function NormalApproxUniformStationary(p)

    mean_std = [
        0.0       1.04197
        -0.529625  0.8581
        0.0       0.622219
        -0.263777  0.557675
        0.0       0.475386
        -0.174502  0.440521
        0.0       0.397355
        -0.129997  0.374798
        0.0       0.347631
        -0.103499  0.331561
    ]

    μ₀ = Float64[]
    σ₀ = Float64[]
    for pₗ in p
        μ₀ = [μ₀;mean_std[1:pₗ,1]]
        σ₀ = [σ₀;mean_std[1:pₗ,2]]
    end

    return μ₀, PDMat(diagm(σ₀.^2))

end

""" 
    MultiSARtoReg(θ::Vector, p::Vector, s::Vector, activeLags; ztrans = "monahan", 
        negative_signs = true) 

Takes a vector `θ` of length sum(p) with unrestricted AR/SAR coefficients and returns the *non-zero* coefficients (determined by `activeLags`) in the product polynomial  
Πₗ(1 - ϕₗ₁B^(sₗ) - ϕₗ₂B^(2*sₗ) - ....) = (1 - ϕ̃₁B - ϕ̃₂B² - ....) 
where the parameters in each AR polynomial is restricted to stability region. 
""" 
function MultiSARtoReg(θ, p, s, activeLags; ztrans = "monahan", negative_signs = true)

    ARpolynomials = Array{Polynomial}(undef, length(s))
    if sum(p) == 1 # only on parameter in the whole model
        ϕₗ = arma_reparam(θ; ztrans = ztrans, negative_signs = negative_signs)[1] .+ eps()
        ARseasonpolymat = [zeros(s[1]-1, p[1]);-ϕₗ']; 
        ARpolynomials[1] = Polynomial([1;ARseasonpolymat[:]], :z)
    else
        count = 0
        for l in 1:length(s)
            ϕₗ = arma_reparam(θ[(count + 1):(count + p[l])]; ztrans = ztrans, 
                negative_signs = negative_signs)[1] .+ eps()
            ARseasonpolymat = [zeros(s[l]-1, p[l]);-ϕₗ']; 
            ARpolynomials[l] = Polynomial([1;ARseasonpolymat[:]], :z)
            count = count + p[l]
        end
    end
    ϕ̃ = -coeffs(prod(ARpolynomials))[2:end]

    return ϕ̃[activeLags]

end