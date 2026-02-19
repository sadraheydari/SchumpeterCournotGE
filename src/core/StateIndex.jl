# ================================================================
# TupleIndexMap  —  combinatorial order
# ================================================================

"""
    tuple_to_index(t::Vector{Int}) -> Int

Convert a **sorted** tuple `t` to its 1-based combinatorial index
using the combinatorial number system.

# Arguments
- `t`: A sorted vector of non-negative integers.
"""
function tuple_to_index(t::Vector{Int})
    n = length(t)
    s = [t[i] + (i - 1) for i in 1:n]
    rank = sum(binomial(s[i] - 1, i) for i in 1:n)
    return rank + 1
end


"""
    TupleIndexMap

Precomputed bidirectional mapping between sorted tuples and their
1-based combinatorial indices.

Built once via [`build_lookup`](@ref) and reused throughout the code.
Index order follows the combinatorial number system.

# Fields
- `tuple_to_idx`: Maps a sorted `Vector{Int}` tuple to its 1-based index.
- `idx_to_tuple`: Maps a 1-based index back to its sorted tuple.
- `n`:            Tuple length.
- `m`:            Upper bound of tuple values (values drawn from `1:m`).
"""
struct TupleIndexMap
    tuple_to_idx::Dict{Vector{Int}, Int}
    idx_to_tuple::Vector{Vector{Int}}
    n::Int
    m::Int
end


"""
    build_lookup(n::Int, m::Int) -> TupleIndexMap

Build a [`TupleIndexMap`](@ref) for all sorted tuples of length `n`
with values drawn from `1:m`, using `with_replacement_combinations`
to enumerate all tuples directly.

The total number of such tuples is `binomial(m + n - 1, n)`.
Call this function **once** at startup and reuse the result.

# Arguments
- `n`: Tuple length.
- `m`: Upper bound of tuple values.
"""
function build_lookup(n::Int, m::Int)
    total = binomial(m + n - 1, n)
    idx_to_tuple = Vector{Vector{Int}}(undef, total)
    tuple_to_idx = Dict{Vector{Int}, Int}()

    for (idx, combo) in enumerate(with_replacement_combinations(1:m, n))
        t = collect(combo)
        idx_to_tuple[idx] = t
        tuple_to_idx[t] = idx
    end

    return TupleIndexMap(tuple_to_idx, idx_to_tuple, n, m)
end


# ================================================================
# @sget  —  sort-guarded lookup, works for both map types
# ================================================================

"""
    @sget expr

Sort-guarded index lookup. Given an indexing expression of the form
`container[key]`, checks whether `key` is sorted before performing
the lookup. If not, sorts it first.

Use this macro at call sites where the tuple may arrive unsorted.
Use plain indexing `container[key]` when the tuple is guaranteed sorted.
Works with both [`TupleIndexMap`](@ref) and [`BandedIndexMap`](@ref).
"""
macro sget(expr)
    @assert expr.head == :ref "Expected an indexing expression like map.tuple_to_idx[tuple]"
    dict = expr.args[1]
    key  = expr.args[2]
    return quote
        let k = $(esc(key))
            $(esc(dict))[issorted(k) ? k : sort(k)]
        end
    end
end