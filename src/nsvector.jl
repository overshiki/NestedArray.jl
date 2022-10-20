
# const Maybe{T} = Union{T, Nothing}
using EasyMonad
import EasyMonad.(>>)
"""
I do consider in the situations below, the alias of length function(just the pythonic `len` function) would be convenient.
"""
len(vs::Vector{T}) where T = length(vs)
len(d::Dict) = length(d)
len(s::String) = length(s)
len(t::Tuple) = length(t)

# """
# Monad bind: M [a] -> ([a] -> b) -> M b
# this one is really convenient
# """
# maybebind(x::Maybe{T}, f::Function) where T = begin
#     x isa Nothing && return x 
#     return f(x)
# end

const ViewType{T} = SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}
const VVector{T} = Union{Vector{T}, ViewType{T}}

"""
will this one be cheaper than the previous one?
Yes, but still not as good as in a simple for loop... since in julia, deep recursive may resulted in stackoverflow
"""
(find_item_index(vs::VVector{T}, item::T, index::Int)::Maybe{Int}) where T = begin 
    length(vs)==0 && return nothing
    vs[1]==item && return index 
    return find_item_index(view(vs, 2:length(vs)), item, index+1)
end

(find_item_index(vs::Vector{T}, item::T, index::Int, ::Val{:old})::Maybe{Int}) where T = begin 
    length(vs)==0 && return nothing
    vs[1]==item && return index 
    return find_item_index(vs[2:end], item, index+1)
end

(find_unique_index(vs::VVector{T}, item::T)::Int) where T = begin 
    maybe_index = find_item_index(vs, item, 1)
    @assert (maybe_index >> i->find_item_index(view(vs, i+1:length(vs)), item, 1)) isa Nothing 
    return maybe_index
end

(find_index(vs::Vector{T}, v::T, ::Val{:recursive})::Maybe{Int}) where T = find_item_index(vs, v, 1)
"""
To find the index of an item in a vector. If it does not exist in the vector, just return nothing
by default, I recommend using this simple loop version.
"""
(find_index(vs::Vector{T}, v::T)::Maybe{Int}) where T = begin 
    for (index, iv) in enumerate(vs) 
        iv==v && return index
    end
    return nothing
end


######## utils for NSVector ########
(++)(xlist::Vector{T}, ylist::Vector{T}) where T = begin 
    clist = T[]
    append!(clist, xlist)
    append!(clist, ylist)
    return clist
end 



unsqueeze(a::Array) = begin 
    nsize = foldl(append!, size(a); init=[1])
    return reshape(a, Tuple(nsize))
end
fullstack(vs::Vector{T}) where T<:Vector = begin
    return reduce(vcat, map(unsqueeze, map(fullstack, vs)))
end
fullstack(vs::Vector{T}) where T<:Real = vs

concat(alist::Vector{Vector{T}}) where T = foldl(++, alist; init=T[])
concat(alist::Vector{T}) where T<:Real = alist

Empty(ns::Vector{T}) where T<:Real = T[]
(Empty(ns::Vector{Vector{T}})::Vector{Vector{T}}) where T<:Real = map(Empty, ns)


nvsize!(ns::Vector{T}, svec::Vector{Int}) where T = begin 
    push!(svec, len(ns))
    T <: Real && return svec
    nvsize!(ns[1], svec)
end
(nvsize(ns::Vector{T})::Tuple) where T = begin 
    svec = nvsize!(ns, Int[])
    Tuple(svec)
end

hsplit(a::Array)::Vector{Array} = begin 
    asize = size(a)
    v = asize[1]
    map(1:v) do i 
        return Array(selectdim(a, 1, i))
    end
end

from_array(a::Array) = begin 
    asize = size(a)
    length(asize)==1 && return a
    v = asize[1]
    map(1:v) do i 
        return from_array(Array(selectdim(a, 1, i)))
    end
end

squeeze(vs::Vector{T}) where T = begin 
    length(nvsize(vs))==1 && return vs
    len(vs)==1 && return squeeze(vs[1])
    map(vs) do v 
        return squeeze(v)
    end
end

cast(vs::Vector{T}, t::Type{T2}) where {T, T2<:Real} = begin 
    T <: Real && return map(t, vs)
    return map(v->cast(v, t), vs) 
end

const AtLeast2D{T} = Vector{Vector{T}}
transpose(ns::AtLeast2D{T}) where T = begin 
    inner_len = nvsize(ns)[2]
    map(1:inner_len) do i 
        map(1:len(ns)) do j 
            ns[j][i]
        end |> concat
    end
end

#TODO: fix this, originally, the inner most vector is always [x] in MaybeTensor.jl, I need to modify this
transpose(ns::AtLeast2D{T}, startIndex::Int) where T = begin 
    startIndex==1 && return transpose(ns)
    return map(x->transpose(x, startIndex-1), ns)
end

struct StartIndex 
    index::Int
end

"""
    (1, 2, 3) -> (2, 3, 1) -> (1, 2), (2, 3)
    (1, 2, 3) -> (3, 1, 2) -> (2, 3), (1, 2)
    (1, 2, 3) -> (3, 2, 1) -> (1, 2), (2, 3), (1, 2)
"""
transpose_schedule(orders::Vector{Int}, schedule::Vector{StartIndex}) = begin
    len(orders)==0 && return schedule
    o, os = orders[end], orders[1:end-1]    
    schedule = schedule ++ map(StartIndex, o:len(orders)-1)

    len(os)==0 && return schedule

    nos = map(os) do oi 
        oi > o && return oi -1 
        oi < o && return oi
    end
    transpose_schedule(nos, schedule)
end
transpose_schedule(orders::Vector{Int}) = transpose_schedule(orders, StartIndex[])


(transpose(ns::AtLeast2D{T}, schedule::Vector{StartIndex})::AtLeast2D{T}) where T = begin
    return foldl((ns, s)->transpose(ns, s.index), schedule; init=ns)
end
(transpose(ns::AtLeast2D{T}, targetVec::Vector{Int})::AtLeast2D{T}) where T = begin
    return transpose(ns, transpose_schedule(targetVec))
end
