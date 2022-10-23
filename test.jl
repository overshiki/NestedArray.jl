include("src/nsvector.jl")
# using NestedArray

test() = begin 
    a = rand(Float32, (4,3,2))
    display(a)
    vs = from_array(a)
    @show map(nvsize, vs)
    @show vs |> nvsize
    nvs = vs |> fullstack 

    nvs |> display

    @show all(a .== nvs)

    a = rand(Float32, (10, 1, 1, 20))
    a = from_array(a)
    # @show length(size(a))
    a = squeeze(a)
    @show nvsize(a)

    # @show cast(a, Float64) |> nvsize
    a |> fullstack |> display

    a = a |> transpose
    @show nvsize(a)

    a |> fullstack |> display

    a = rand(Float32, (4,3,2))
    vs = from_array(a)


    # vs = transpose(vs, [3,2,1])
    @show nvsize(vs)
    # map(x->print(size(x)), vs)
end 

test2() = begin 
    a = rand(10000)
    a[99] = 1.0
    @show @allocated find_item_index(a, 1.0, 1)
    @show @allocated find_item_index(a, 1.0, 1, Val(:old))
    @show @allocated find_index(a, 1.0, Val(:recursive))
    @show @allocated find_index(a, 1.0)
    @show find_unique_index(a, 1.0)
end


test_nvbroadcast() = begin 
    sa = collect(1:10)
    ns = rand(5:15, 10)
    ssb = map(n->collect(1:n), ns)
    # @show sa, ssb
    func = BinaryFunction{Int, Int, Int}((x,y)->x+y)
    ssb = nvbroadcast(sa, ssb, func)
    @show ssb
end

# test()
# test2()
test_nvbroadcast()
