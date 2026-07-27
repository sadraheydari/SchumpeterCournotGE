"""
    SymStateArrays

Dense storage over a discretised state space in which all but the first
component of the state are exchangeable.

# The state

A state is a length-`n` vector of **grid indices**

    x = (g, y₁, y₂, …, y_{n-1})

where

  * `g = x[1] ∈ 1:k` indexes the first grid, `G = {g_1, …, g_k}`;
  * `y = x[2:end] ∈ (1:k_y)^(n-1)` indexes a second (possibly different,
    usually coarser) grid and is **permutation-symmetric**: the state is
    unchanged by reordering the `y`ᵢ.

By default `k_y = k`, i.e. every component lives on the same grid.

Indices are positions in `1:k` / `1:k_y`, *not* the grid values themselves.
Keep the values in an ordinary vector alongside the array:

```julia
G = collect(range(0.0, 10.0; length = 100))
V = StateArray(4, length(G))
value_at_gridpoint = G[g]
```

# What the type does

Only one slot is stored per equivalence class of `y`, and every lookup
canonicalises its argument by sorting. Hence

    X[g, y...] == X[g, permutation(y)...]

holds *structurally*: the two expressions resolve to the same memory, so a
write through one ordering is visible through every other. There is no way
for permuted duplicates to drift apart, because they do not exist.

# Storage layout

`X.data` is a `k × nsymstates(n, k_y)` matrix, where

    nsymstates(n, k_y) = C(k_y + n - 2, n - 1)

is the number of multisets of size `n-1` drawn from `1:k_y`. The row is `g`
and the column is the colexicographic rank of the sorted `y` (see
`stateindex`). Julia is column-major, so **`g` is the contiguous index**:
walking `g` at fixed `y` is sequential, walking `y` at fixed `g` strides by
`k` elements. Orient your Bellman sweep accordingly — if the expectation
runs over `y` rather than `g`, consider storing the transpose.

`n == 1` is the degenerate case: there is no `y` part, the column count
collapses to 1, and the object is a `k × 1` matrix indexed by `X[g]`.
`vec(X)` gives a flat view.

# Cost

Indexing is `O(n)` and **independent of `k`**: an unrolled insertion sort of
`n-1` integers followed by `n-1` lookups into a binomial table of
`(k_y + n - 1) × (n - 1)` `Int`s. That table is a few kilobytes even for
large grids, so it stays resident in L1. Lookups allocate nothing and, with
`@inbounds`, compile to little more than the arithmetic.

Memory is the binding constraint, not speed. Total elements is
`nstates(n, k, k_y) = k * nsymstates(n, k_y)`; at `k = k_y = 100`:

| `n` | elements      | `Float64` |
|:----|:--------------|:----------|
| 3   | 505,000       | 4 MB      |
| 4   | 17,170,000    | 137 MB    |
| 5   | 442,127,500   | 3.5 GB    |

The jump from `n=4` to `n=5` is steep, and value-function iteration needs at
least two arrays live at once. Coarsening the `y` grid is the cheapest
remedy: at `n = 5, k = 100`, dropping to `k_y = 50` costs 234 MB instead of
3.5 GB. Call `nstates` before allocating.

# Example

```julia
using .SymStateArrays

V = StateArray(4, 100)          # Float64, 100×171_700, all components on 1:100
V = StateArray(4, 100, 50)      # g on 1:100, the three y's on 1:50
V = StateArray{Float32}(4, 100) # half the memory

V[7, 3, 9, 3] = 1.5
V[7, 9, 3, 3]                   # 1.5 — same slot
stateindex(V, (7, 9, 3, 3))     # (7, j) — canonical position in V.data
```
"""
module SymStateArrays

export StateArray, stateindex, nsymstates, nstates,
       gridsize, ygridsize, statelength

# =====================================================================
#  Allocation-free insertion sort on tuples.
#
#  `n` is small (typically ≤ 10), so an insertion sort written as tuple
#  recursion is the right tool: the compiler unrolls it completely, it
#  touches no heap, and it beats any general sorting routine at this size.
#  This is the same construction TupleTools.jl uses; it is inlined here to
#  keep the file dependency-free.
#
#  Both branches of `_tinsert` return an `NTuple` of the same length and
#  element type, which is what makes the recursion type-stable.
# =====================================================================
@inline _tsort(t::Tuple{})    = ()
@inline _tsort(t::Tuple{Any}) = t
@inline function _tsort(t::Tuple)
    @inline _tinsert(t[1], _tsort(Base.tail(t)))
end

@inline _tinsert(v, ::Tuple{}) = (v,)
@inline function _tinsert(v, t::Tuple)
    v <= t[1] ? (v, t...) : (t[1], _tinsert(v, Base.tail(t))...)
end

# =====================================================================
#  The array type
# =====================================================================

"""
    StateArray{T,N,M,A}

Dense array over states `x = (g, y…)` with permutation-symmetric `y`.
See the module docstring for the storage scheme.

Type parameters:

  * `T` — element type;
  * `N` — length of a state, i.e. `n`;
  * `M` — `N - 1`, the length of the symmetric part `y`;
  * `A` — the type of the backing matrix.

`N` and `M` are type parameters rather than fields so that the sort and the
ranking loop are unrolled at compile time.

Fields:

  * `data`  — the `k × nsymstates(n, k_y)` backing matrix;
  * `k`     — size of the grid for `x[1]`;
  * `ky`    — size of the grid for `x[2:end]`;
  * `binom` — table with `binom[a+1, i] == binomial(a, i)`, used to rank
    multisets in `O(n)`.
"""
struct StateArray{T,N,M,A<:AbstractMatrix{T}}
    data::A
    k::Int
    ky::Int
    binom::Matrix{Int}
end

"""
    nsymstates(n, ky) -> Int
    nsymstates(X)     -> Int

Number of distinct sorted `y`-vectors: multisets of size `n-1` drawn from
`1:ky`, equal to `C(ky + n - 2, n - 1)`. This is the number of columns of
`X.data`. For `n == 1` there is no `y` part and the result is `1`.

```julia
nsymstates(4, 100)   # 171_700
nsymstates(1, 100)   # 1
```

See also [`nstates`](@ref).
"""
nsymstates(n::Integer, ky::Integer) =
    n == 1 ? 1 : binomial(ky + n - 2, n - 1)
nsymstates(X::StateArray) = size(X.data, 2)

"""
    nstates(n, k, ky = k) -> Int
    nstates(X)            -> Int

Total number of stored states, `k * nsymstates(n, ky)`. Multiply by
`sizeof(T)` for the memory footprint of one array.

Worth calling before you allocate — the count grows steeply in `n`:

```julia
nstates(4, 100)          # 17_170_000    → 137 MB as Float64
nstates(5, 100)          # 442_127_500   → 3.5 GB as Float64
nstates(5, 100, 50)      # 29_282_500    → 234 MB as Float64
```
"""
nstates(n::Integer, k::Integer, ky::Integer = k) = k * nsymstates(n, ky)
nstates(X::StateArray) = length(X.data)

"""
    StateArray{T}(n, k, ky = k)
    StateArray(n, k, ky = k)        # T = Float64

Zero-initialised array over states of length `n`, with `x[1] ∈ 1:k` and the
permutation-symmetric part `x[2:end] ∈ (1:ky)^(n-1)`.

Pass `ky < k` when the exchangeable components need less resolution than the
first one; this is usually the cheapest way to control memory, since the
column count grows like `ky^(n-1)`.

Construction is `O(k_y · n)` — it fills a small binomial table — so build
these once outside your iteration loop.

```julia
V = StateArray(4, 100)              # 100 × 171_700 Float64
P = StateArray{Int}(4, 100)         # policy indices, same shape
W = StateArray{Float32}(5, 100, 50) # coarse y grid, single precision
```

Throws `ArgumentError` if `n < 1`, `k < 1` or `ky < 1`.
"""
function StateArray{T}(n::Integer, k::Integer, ky::Integer = k) where {T}
    n  >= 1 || throw(ArgumentError("n must be ≥ 1 (got $n)"))
    k  >= 1 || throw(ArgumentError("k must be ≥ 1 (got $k)"))
    ky >= 1 || throw(ArgumentError("ky must be ≥ 1 (got $ky)"))
    M    = Int(n) - 1
    nsym = nsymstates(n, ky)

    # binom[a+1, i] == binomial(a, i), for a ∈ 0:(ky+M-1), i ∈ 1:M.
    # `binomial` returns 0 whenever a < i, which is exactly what the ranking
    # formula needs for the leading entries of a multiset.
    B = Matrix{Int}(undef, Int(ky) + M, M)
    @inbounds for i in 1:M, a in 0:(Int(ky) + M - 1)
        B[a + 1, i] = binomial(a, i)
    end

    data = zeros(T, Int(k), nsym)
    StateArray{T,Int(n),M,typeof(data)}(data, Int(k), Int(ky), B)
end
StateArray(n::Integer, k::Integer, ky::Integer = k) =
    StateArray{Float64}(n, k, ky)

# =====================================================================
#  Basic queries
# =====================================================================

"""
    gridsize(X) -> Int

Size of the grid for the first component, `x[1] ∈ 1:gridsize(X)`.
Equals `size(X.data, 1)`.
"""
gridsize(X::StateArray) = X.k

"""
    ygridsize(X) -> Int

Size of the grid for the symmetric components, `x[i] ∈ 1:ygridsize(X)` for
`i ≥ 2`. Equals `gridsize(X)` unless a separate `ky` was given.
"""
ygridsize(X::StateArray) = X.ky

"""
    statelength(X) -> Int

Length `n` of a state vector, including the first component.
"""
statelength(::StateArray{T,N}) where {T,N} = N

Base.eltype(::StateArray{T}) where {T} = T

"""
    size(X)   -> (k, nsymstates(X))
    length(X) -> number of stored states

These describe the *storage*, `X.data`, not the state space before
symmetry reduction.
"""
Base.size(X::StateArray)   = size(X.data)
Base.length(X::StateArray) = length(X.data)

Base.fill!(X::StateArray, v) = (fill!(X.data, v); X)

"""
    copy(X) -> StateArray

Independent copy. The backing matrix is copied; the binomial table is
shared, since it is read-only and depends only on `(n, ky)`.
"""
Base.copy(X::StateArray{T,N,M}) where {T,N,M} =
    StateArray{T,N,M,typeof(X.data)}(copy(X.data), X.k, X.ky, X.binom)

"""
    vec(X) -> Vector view

Flat view of the storage, sharing memory with `X.data`. The natural form
when `n == 1`, where the array is `k × 1`.
"""
Base.vec(X::StateArray) = vec(X.data)

function Base.show(io::IO, X::StateArray{T,N}) where {T,N}
    print(io, "StateArray{", T, "}(n=", N, ", k=", X.k)
    X.ky == X.k || print(io, ", ky=", X.ky)
    print(io, ") — ", size(X.data, 1), "×", size(X.data, 2), " = ",
              length(X.data), " states")
end

# =====================================================================
#  The index map: an unordered y ⟶ a column
#
#  Ranking multisets. Let 1 ≤ y₁ ≤ … ≤ y_M ≤ ky be the sorted symmetric
#  part. The shift
#
#      cᵢ = yᵢ + i - 1
#
#  turns it into a *strictly* increasing sequence 1 ≤ c₁ < … < c_M ≤ ky+M-1,
#  i.e. an M-subset of 1:(ky+M-1). Multisets of size M from ky values and
#  M-subsets of ky+M-1 values are therefore in bijection — the standard
#  stars-and-bars correspondence, which is why the count is C(ky+M-1, M).
#
#  Such a subset is ranked colexicographically by
#
#      rank₀ = Σᵢ C(cᵢ - 1, i)
#
#  which is a bijection onto 0:(C(ky+M-1, M) - 1). Adding 1 gives a valid
#  Julia column index. Every term is a table lookup, so the whole map costs
#  M additions — no division, no search, no hashing.
# =====================================================================

"""
    _symcol(X, ysorted) -> Int

Column index of the already-sorted symmetric part `ysorted`. Internal: the
sortedness precondition is not checked, and violating it silently returns
the wrong column. Use [`stateindex`](@ref) instead.
"""
@inline function _symcol(X::StateArray{T,N,M}, ysorted::NTuple{M,Int}) where {T,N,M}
    B = X.binom
    r = 1
    @inbounds for i in 1:M
        r += B[ysorted[i] + i - 1, i]     # binomial(cᵢ - 1, i)
    end
    return r
end

"""
    _checkstate(X, g, y)

Throw `BoundsError` unless `g ∈ 1:k` and every `yᵢ ∈ 1:ky`. Internal;
called from within `@boundscheck`, so `@inbounds` elides it.
"""
@inline function _checkstate(X::StateArray{T,N,M}, g::Int,
                             y::NTuple{M,Int}) where {T,N,M}
    (1 <= g <= X.k) || throw(BoundsError(X, (g, y...)))
    for v in y                            # homogeneous tuple ⇒ unrolled
        (1 <= v <= X.ky) || throw(BoundsError(X, (g, y...)))
    end
    return nothing
end

"""
    stateindex(X, x) -> (i, j)

Canonical position of state `x` in `X.data`: row `i = x[1]`, column `j` the
rank of the multiset `x[2:end]`. The symmetric components may be given in
any order, and permutations of one another map to the same `(i, j)`.

`x` may be a `Tuple` or an `AbstractVector` of integers, of length
`statelength(X)`.

Use this when you want the storage coordinates themselves — for instance to
read and write a slot repeatedly without re-ranking, or to record positions
in a transition matrix:

```julia
i, j = stateindex(V, x)
V.data[i, j] += δ
```

Throws `DimensionMismatch` if `length(x) != statelength(X)`, and
`BoundsError` if any component is off its grid. Both checks are removed by
`@inbounds`, which then assumes the state is valid — an out-of-range index
will then read or write the wrong slot rather than error.
"""
@inline function stateindex(X::StateArray{T,N,M}, x) where {T,N,M}
    @boundscheck length(x) == N || throw(DimensionMismatch(
        "state has length $(length(x)), expected $N"))
    g = Int(@inbounds x[1])
    y = ntuple(i -> Int(@inbounds x[i + 1]), Val(M))
    @boundscheck _checkstate(X, g, y)
    return (g, _symcol(X, _tsort(y)))
end

# =====================================================================
#  Indexing
#
#  Three equivalent spellings, all canonicalising:
#
#      X[g, y₁, …, y_{n-1}]     varargs — needs exactly n integers
#      X[x]                     x a Tuple or AbstractVector of length n
#      X(x...) / X(x)           callable form, handy when passing X around
#
#  `StateArray` deliberately does *not* subtype `AbstractArray`: its
#  indexing semantics are not those of an array (arguments are canonicalised,
#  and a vector argument means one state rather than a gather), so inheriting
#  the `AbstractArray` fallbacks would create more confusion than value.
#  Reach through to `X.data` when you want ordinary array operations —
#  broadcasting, reductions, `copyto!` — on the storage.
# =====================================================================

"""
    X[x]
    X[g, y₁, …, y_{n-1}]
    X(x) / X(g, y₁, …, y_{n-1})

Read the value at state `x`. The symmetric components may be in any order.

```julia
V[7, 3, 9, 3] == V[7, 9, 3, 3] == V[(7, 3, 3, 9)] == V[[7, 9, 3, 3]]
```

Wrap hot loops in `@inbounds` once you trust the indices.
"""
@inline function Base.getindex(X::StateArray,
                               x::Union{Tuple,AbstractVector{<:Integer}})
    i, j = stateindex(X, x)
    return @inbounds X.data[i, j]
end

"""
    X[x] = v
    X[g, y₁, …, y_{n-1}] = v

Write `v` at state `x`. Since permutations share a slot, this overwrites the
whole equivalence class.
"""
@inline function Base.setindex!(X::StateArray, v,
                                x::Union{Tuple,AbstractVector{<:Integer}})
    i, j = stateindex(X, x)
    @inbounds X.data[i, j] = v
    return v
end

@inline Base.getindex(X::StateArray{T,N}, x::Vararg{Integer,N}) where {T,N} =
    X[x]
@inline Base.setindex!(X::StateArray{T,N}, v, x::Vararg{Integer,N}) where {T,N} =
    (X[x] = v)

@inline (X::StateArray{T,N})(x::Vararg{Integer,N}) where {T,N} = X[x]
@inline (X::StateArray)(x::Union{Tuple,AbstractVector{<:Integer}}) = X[x]

end # module