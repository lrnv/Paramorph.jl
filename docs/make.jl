using Paramorph
using Documenter

DocMeta.setdocmeta!(Paramorph, :DocTestSetup, :(using Paramorph); recursive=true)

makedocs(;
    modules=[Paramorph],
    authors="Oskar Laverny <oskar.laverny@univ-amu.fr> and contributors",
    sitename="Paramorph.jl",
    format=Documenter.HTML(;
        canonical="https://lrnv.github.io/Paramorph.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutoriel" => "tutorial.md",
    ],
)

deploydocs(;
    repo="github.com/lrnv/Paramorph.jl",
    devbranch="main",
)
