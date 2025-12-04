# TVMultiSARMA

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mattiasvillani.github.io/TVMultiSARMA.jl/dev/)
[![Build Status](https://github.com/mattiasvillani/TVMultiSARMA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mattiasvillani/TVMultiSARMA.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mattiasvillani/TVMultiSARMA.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mattiasvillani/TVMultiSARMA.jl)

## Description

This package implements Gibbs sampling for Bayesian inference in time-varying multi-seasonal ARMA (TV-Multi-SARMA) models using the Sequential Monte Carlo (SMC) samplers in [SMCsamplers.jl](https://github.com/mattiasvillani/SMCsamplers.jl) and the Gibbs sampling for dynamic global-local shrinkage priors in [DynamicGlobalLocalShrinkage.jl](https://github.com/mattiasvillani/DynamicGlobalLocalShrinkage.jl).

## Installation
Install from the Julia package manager (via Github) by typing `]` in the Julia REPL:
```
] add git@github.com:mattiasvillani/TVMultiSARMA.jl.git
```
## Multi-seasonal AR model

The TVSAR(p,P) with $p$ regular lags and $P$ seasonal lags with a single seasonality $s$ is
```math
\begin{equation*}
    \phi_t(L)\Phi_t(L^s)y_t = \varepsilon_t , \quad \varepsilon_t \sim N(0,\sigma_t^2)  
\end{equation*}
```
where $\phi_t(L) = 1 - \phi_{1t} L - \phi_{2t} L^2 - \ldots - \phi_{pt} L^p$ is the regular AR polynomial and $\Phi_t(L^s) = 1 - \Phi_{1t} L^s - \Phi_2 L^{2s} - \ldots - \Phi_{Pt} L^{Ps}$ is the seasonal AR polynomial.

The TVMultiSARMA.jl package allows any number of seasonalities $s_1, s_2, \ldots, s_K$ by including additional seasonal AR polynomials. 

The AR parameters $\phi_{jt}$ and $\Phi_{jt}$ can optionally be restricted so that the process is stable at every $t$ by the composite map  

```math
\boldsymbol{\theta}_t \rightarrow \boldsymbol{r}_t \rightarrow \boldsymbol{\phi}_t, 
```
where the unrestricted parameters 𝛉ₜ in ℝᵖ are first mapped to the partial autocorrelations 𝐫ₜ in [−1, 1]ᵖ which are then mapped to the stable AR parameters 𝛟ₜ. The map from 𝛉ₜ to 𝐫ₜ can take many forms, for example, the Monahan transformation
 
```math
 r_{jt} = \frac{\theta_{jt}}{\sqrt{1 + \theta_{jt}^2}}.
```

The unrestricted parameters 𝛉ₜ evolve over time following independent dynamic shrinkage process (DSP) priors. For example, for a single AR parameter $\theta_t$ the DSP prior is

```math
\begin{align*}
\theta_t &= \theta_{t-1} + \nu_t , \quad \nu_t \sim N(0,\exp(h_t)) \\
h_t &= \mu + \kappa(h_{t-1} -\mu) + \eta_t, \quad \eta_t \sim Z(\alpha,\beta,0,1)  
\end{align*}
```
where $Z(\alpha,\beta,0,1)$ is the Z-distribution with parameters $\alpha=\beta=1/2$.
The DSP prior has a global log-variance $\mu$ that determines the overall degree of 
time-variation, and a local log-variance component $\eta_t$ that allows for 
large changes in the parameter innovation variance, changes that can be 
persistent over time due to the AR(1) structure in $h_t$.

The TVMultiSARMA.jl allows for a stochastic volatility model for the measurement variance $\sigma_t^2$

```math
\begin{equation*}
    \log \sigma_t^2 = \log \sigma_{t-1}^2 + \xi_t, \quad \xi_t \sim N(0,\sigma_\xi^2)  
\end{equation*}
```
Future versions will include dynamic shrinkage process prior for $\sigma_t^2$.

## Limitations and future extensions

The TVMultiSARMA.jl package is limited to time-varying AR models and Bayesian inference using the conditional likelihood. The stochastic volatility model for the measurement errors uses a homoscedastic Gaussian parameter evolation for $\log\sigma_t$. Future versions will extend this by adding:

- moving average MA and seasonal MA components
- the exact likelihood
- global-local shrinkage priors for the stochastic volatility model

The current package is not optimized for speed, and is rather sloppy with memory allocations and type instabilities. Future versions will include speed optimizations.

## Examples

See the documentation and the examples folder for usage and illustrations:

- [Monthly US industrial production data 1919-2024](https://github.com/mattiasvillani/TVMultiSARMA.jl/blob/main/examples/usip/script.jl)

## References

- Fagerberg, G., Villani, M., & Kohn, R. (2025). Time-Varying Multi-Seasonal AR Models. [arXiv](https://arxiv.org/abs/2409.18640)