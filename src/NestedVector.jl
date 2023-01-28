module NestedVector
include("nsvector.jl")
export find_index, find_unique_index, (++), 
        fullstack, concat, Empty, nvsize, 
        squeeze, cast, 
        hsplit, nvbroadcast, foldvector,
        from_array

end # module
