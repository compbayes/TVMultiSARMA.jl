# TVMultiSARMA

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mattiasvillani.github.io/TVMultiSARMA.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mattiasvillani.github.io/TVMultiSARMA.jl/dev/)
[![Build Status](https://github.com/mattiasvillani/TVMultiSARMA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mattiasvillani/TVMultiSARMA.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mattiasvillani/TVMultiSARMA.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mattiasvillani/TVMultiSARMA.jl)

## Description

This package implements Gibbs sampling for Bayesian inference in time-varying multi-seasonal ARMA (TV-Multi-SARMA) models using the sequential Monte Carlo (SMC) samplers in [SMCsamplers.jl](https://github.com/mattiasvillani/SMCsamplers.jl) and the Gibbs sampling for dynamic global-local shrinkage priors in [DynamicGlobalLocalShrinkage.jl](https://github.com/mattiasvillani/DynamicGlobalLocalShrinkage.jl).

## Installation
Install from the Julia package manager (via Github) by typing `]` in the Julia REPL:
```
] add git@github.com:mattiasvillani/TVMultiSARMA.jl.git
```

## Example

TODO: Add example here