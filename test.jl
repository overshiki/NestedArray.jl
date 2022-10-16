# include("src/nsvector.jl")
using NestedArray

a = rand(Float32, (4,3,2))
display(a)
vs = from_array(a)
@show map(nvsize, vs)
@show vs |> nvsize
nvs = vs |> fullstack 

nvs |> display

@show all(a .== nvs)

# map(x->print(size(x)), vs)
