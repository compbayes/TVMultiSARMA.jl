using TVMultiSARMA
using Documenter

DocMeta.setdocmeta!(TVMultiSARMA, :DocTestSetup, :(using TVMultiSARMA); recursive=true)

makedocs(;
    modules=[TVMultiSARMA],
    authors="Mattias Villani, Ganna Fagerberg",
    sitename="TVMultiSARMA.jl",
    format=Documenter.HTML(;
        canonical="https://mattiasvillani.github.io/TVMultiSARMA.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mattiasvillani/TVMultiSARMA.jl",
    devbranch="main",
)
