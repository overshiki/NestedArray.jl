
const Maybe{T} = Union{T, Nothing}

len(vs::Vector{T}) where T = length(vs)
len(d::Dict) = length(d)


"""Monad bind: M [a] -> ([a] -> b) -> M b"""
maybebind(x::Maybe{T}, f::Function) where T = begin
    x isa Nothing && return x 
    return f(x)
end

(find_item_index(vs::Vector{T}, item::T, index::Int)::Maybe{Int}) where T = begin 
    length(vs)==0 && return nothing
    vs[1]==item && return index 
    return find_item_index(vs[2:end], item, index+1)
end
(find_unique_item_index(vs::Vector{T}, item::T)::Int) where T = begin 
    maybe_index = find_item_index(vs, item, 1)
    @assert maybebind(maybe_index, i->find_item_index(vs[i+1:end], item, 1)) isa Nothing 
    return maybe_index
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

transpose(ns::Vector{Vector{T}}) where T = begin 
    inner_len = nvsize(ns)[2]
    map(1:inner_len) do i 
        map(1:len(ns)) do j 
            ns[j][i]
        end |> concat
    end
end