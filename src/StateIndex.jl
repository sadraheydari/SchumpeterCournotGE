using Combinatorics
using DataStructures   # for Queue in BFS

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
# BandedIndexMap  —  BFS transition order
# ================================================================

"""
    BandedIndexMap

Precomputed bidirectional mapping between sorted tuples and 1-based
indices ordered by BFS over the "+0/+1" transition graph.

States that are reachable from one another in fewer transitions are
assigned closer indices. This produces a banded transition matrix,
which improves cache locality and convergence speed when solving
systems of equations over these states.

BFS starts from `(1, 1, …, 1)` and expands all neighbours obtained
by independently incrementing each element of the tuple by 0 or 1,
keeping only sorted results within bounds.

Built once via [`build_banded_lookup`](@ref) and reused throughout.

# Fields
- `tuple_to_idx`: Maps a sorted `Vector{Int}` tuple to its 1-based index.
- `idx_to_tuple`: Maps a 1-based index back to its sorted tuple.
- `n`:            Tuple length.
- `m`:            Upper bound of tuple values (values drawn from `1:m`).
"""
struct BandedIndexMap
    tuple_to_idx::Dict{Vector{Int}, Int}
    idx_to_tuple::Vector{Vector{Int}}
    n::Int
    m::Int
end


"""
    _neighbors(t::Vector{Int}, m::Int) -> Vector{Vector{Int}}

Return all valid sorted neighbors of tuple `t` under the "+0/+1"
transition: each element can stay the same or increment by 1,
subject to values remaining within `1:m` and the result being sorted.

This is an internal helper for [`build_banded_lookup`](@ref).

# Arguments
- `t`: A sorted tuple (current state).
- `m`: Upper bound of tuple values.
"""
function _neighbors(t::Vector{Int}, m::Int)
    n = length(t)
    neighbors = Vector{Vector{Int}}()
    # iterate over all 2^n combinations of {0,1} increments
    for mask in 0:(2^n - 1)
        delta = [(mask >> (i - 1)) & 1 for i in 1:n]
        neighbor = t .+ delta
        # skip if out of bounds or not sorted (increments can break sortedness)
        if all(neighbor .<= m) && issorted(neighbor)
            push!(neighbors, neighbor)
        end
    end
    return neighbors
end


"""
    build_banded_lookup(n::Int, m::Int) -> BandedIndexMap

Build a [`BandedIndexMap`](@ref) for all sorted tuples of length `n`
with values drawn from `1:m`.

Indices are assigned via BFS over the "+0/+1" transition graph,
starting from `(1, 1, …, 1)`. States close in the graph receive
close indices, producing a banded transition matrix.

The total number of such tuples is `binomial(m + n - 1, n)`.
Call this function **once** at startup and reuse the result.

# Arguments
- `n`: Tuple length.
- `m`: Upper bound of tuple values.
"""
function build_banded_lookup(n::Int, m::Int)
    total = binomial(m + n - 1, n)
    idx_to_tuple = Vector{Vector{Int}}(undef, total)
    tuple_to_idx = Dict{Vector{Int}, Int}()

    origin = ones(Int, n)
    queue  = Queue{Vector{Int}}()
    enqueue!(queue, origin)
    idx = 1

    while !isempty(queue)
        t = dequeue!(queue)
        # skip if already visited
        haskey(tuple_to_idx, t) && continue

        tuple_to_idx[t]   = idx
        idx_to_tuple[idx] = t
        idx += 1

        for neighbor in _neighbors(t, m)
            !haskey(tuple_to_idx, neighbor) && enqueue!(queue, neighbor)
        end
    end

    return BandedIndexMap(tuple_to_idx, idx_to_tuple, n, m)
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