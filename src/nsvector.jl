using EasyMonad
using MonadInterface
import MonadInterface.(>>)
"""
I do consider in the situations below, the alias of length function(just the pythonic `len` function) would be convenient.
"""
const Leaf = Union{Number, String, Nothing}

## overloading length function turns out to be a not good idea
# len(vs::Vector{T}) where T = length(vs)
# len(d::Dict) = length(d)
# len(s::String) = length(s)
# len(t::Tuple) = length(t)


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
fullstack(vs::Vector{T}) where T<:Leaf = vs

concat(alist::Vector{Vector{T}}) where T = foldl(++, alist; init=T[])
concat(alist::Vector{T}) where T<:Leaf = alist

Empty(ns::Vector{T}) where T<:Leaf = T[]
(Empty(ns::Vector{Vector{T}})::Vector{Vector{T}}) where T<:Leaf = map(Empty, ns)


nvsize!(ns::Vector{T}, svec::Vector{Int}) where T = begin 
    push!(svec, length(ns))
    T <: Leaf && return svec
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

(nvbroadcast(vs::Vector{T1}, vvs::Vector{Vector{T2}}, op::BinaryFunction{T1, T2, T3})::Vector{Vector{T3}}) where {T1, T2, T3} = begin 
    @assert length(vs)==length(vvs)
    return map(1:length(vs)) do i
        item = vs[i]
        vitem = vvs[i]
        ivs = map(vitem) do item2
            (item, item2) >> op
        end
        return ivs
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
    length(vs)==1 && return squeeze(vs[1])
    map(vs) do v 
        return squeeze(v)
    end
end

(foldvector(srcv::Vector{T}, targetvv::Vector{Vector{T}}, num::Int)::Vector{Vector{T}}) where T = begin 
    length(srcv)==0 && return targetvv 
    nnum = min(length(srcv), num)
    x, xs = srcv[1:nnum], srcv[nnum+1:end]
    push!(targetvv, x)
    return foldvector(xs, targetvv, num)
end
(foldvector(srcv::Vector{T}, num::Int)::Vector{Vector{T}}) where T = begin 
    targetvv = Vector{T}[]
    return foldvector(srcv, targetvv, num)
end



cast(vs::Vector{T}, t::Type{T2}) where {T, T2<:Leaf} = begin 
    T <: Leaf && return map(t, vs)
    return map(v->cast(v, t), vs) 
end

const AtLeast2D{T} = Vector{Vector{T}}
transpose(ns::AtLeast2D{T}) where T = begin 
    inner_len = nvsize(ns)[2]
    map(1:inner_len) do i 
        map(1:length(ns)) do j 
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
    length(orders)==0 && return schedule
    o, os = orders[end], orders[1:end-1]    
    schedule = schedule ++ map(StartIndex, o:length(orders)-1)

    length(os)==0 && return schedule

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
