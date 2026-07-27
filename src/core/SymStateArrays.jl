"""
    SymStateArrays

Storage over states `x = (g, y)` where `y` is permutation-symmetric.

    x[1]     = g  ∈ 1:k          index into the grid G
    x[2:end] = y  ∈ (1:k)^(n-1)  exchangeable ⇒ stored sorted

Only one slot is kept per equivalence class, so

    X[g, y...] == X[g, permutation(y)...]

holds structurally rather than by convention.

Layout: `X.data` is a `k × C(k+n-2, n-1)` matrix; the column is the
colexicographic rank of the multiset `y`.

`n == 1` is the degenerate case: no `y` part, one column, indexed by `X[g]`.
Use `vec(X)` for a flat view.
"""
module SymStateArrays

export StateArray, stateindex, nsymstates, gridsize, statelength

# ---------------------------------------------------------------------
# allocation-free insertion sort on tuples (unrolled; n is small)
# ---------------------------------------------------------------------
@inline _tsort(t::Tuple{})    = ()
@inline _tsort(t::Tuple{Any}) = t
@inline function _tsort(t::Tuple)
    @inline _tinsert(t[1], _tsort(Base.tail(t)))
end

@inline _tinsert(v, ::Tuple{}) = (v,)
@inline function _tinsert(v, t::Tuple)
    v <= t[1] ? (v, t...) : (t[1], _tinsert(v, Base.tail(t))...)
end

# ---------------------------------------------------------------------
# the array type
#   T = eltype, N = length(x) = n, M = N-1 = length(y)
# ---------------------------------------------------------------------
struct StateArray{T,N,M,A<:AbstractMatrix{T}}
    data::A             # k × nsym
    k::Int              # |G|
    binom::Matrix{Int}  # binom[a+1, i] == binomial(a, i)
end

"""
    nsymstates(n, k)
    nsymstates(X)

Number of distinct sorted `y`-vectors (multisets of size `n-1` drawn from
`1:k`). The total number of stored states is `k * nsymstates(n, k)`.
For `n == 1` this is `1`, so the array is `k × 1`.
"""
nsymstates(n::Integer, k::Integer) =
    n == 1 ? 1 : binomial(k + n - 2, n - 1)
nsymstates(X::StateArray) = size(X.data, 2)

"""
    StateArray{T}(n, k)
    StateArray(n, k)      # Float64

Zero-initialised array over states `x` of length `n`, all components living
on a grid of size `k`, with `x[2:end]` treated as permutation-symmetric.

Check `nsymstates(n, k) * k` before allocating: it grows fast in `n`.
"""
function StateArray{T}(n::Integer, k::Integer) where {T}
    n >= 1 || throw(ArgumentError("n must be ≥ 1 (got $n)"))
    k >= 1 || throw(ArgumentError("k must be ≥ 1 (got $k)"))
    M    = Int(n) - 1
    nsym = nsymstates(n, k)

    # binomial table: rows a = 0:(k+M-1), cols i = 1:M
    B = Matrix{Int}(undef, k + M, M)
    @inbounds for i in 1:M, a in 0:(k + M - 1)
        B[a + 1, i] = binomial(a, i)     # 0 when a < i
    end

    data = zeros(T, Int(k), nsym)
    StateArray{T,Int(n),M,typeof(data)}(data, Int(k), B)
end
StateArray(n::Integer, k::Integer) = StateArray{Float64}(n, k)

# ---------------------------------------------------------------------
# basic queries
# ---------------------------------------------------------------------
Base.eltype(::StateArray{T}) where {T}     = T
Base.size(X::StateArray)                   = size(X.data)
Base.length(X::StateArray)                 = length(X.data)
gridsize(X::StateArray)                    = X.k
statelength(::StateArray{T,N}) where {T,N} = N

Base.fill!(X::StateArray, v) = (fill!(X.data, v); X)
Base.copy(X::StateArray{T,N,M}) where {T,N,M} =
    StateArray{T,N,M,typeof(X.data)}(copy(X.data), X.k, X.binom)

# flat view of the storage; the natural form when n == 1
Base.vec(X::StateArray) = vec(X.data)

function Base.show(io::IO, X::StateArray{T,N}) where {T,N}
    print(io, "StateArray{", T, "}(n=", N, ", k=", X.k, ") with ",
              size(X.data, 1), "×", size(X.data, 2), " = ", length(X.data),
              " stored states")
end

# ---------------------------------------------------------------------
# the index map: sorted y ⟶ column
#
# For 1 ≤ y₁ ≤ … ≤ y_M ≤ k set cᵢ = yᵢ + i - 1, giving a strictly
# increasing M-subset of 1:(k+M-1). Its colex rank is Σᵢ C(cᵢ-1, i),
# a bijection onto 0:(C(k+M-1,M)-1).
# ---------------------------------------------------------------------
@inline function _symcol(X::StateArray{T,N,M}, ysorted::NTuple{M,Int}) where {T,N,M}
    B = X.binom
    r = 1
    @inbounds for i in 1:M
        r += B[ysorted[i] + i - 1, i]     # binomial(cᵢ - 1, i)
    end
    return r
end

"""
    stateindex(X, x) -> (i, j)

Canonical `(row, column)` position of state `x` in `X.data`. The last `n-1`
components of `x` may be given in any order.
"""
@inline function stateindex(X::StateArray{T,N,M}, x) where {T,N,M}
    @boundscheck begin
        length(x) == N || throw(DimensionMismatch(
            "state has length $(length(x)), expected $N"))
        k = X.k
        for v in x
            (1 <= v <= k) || throw(BoundsError(X, x))
        end
    end
    g = Int(@inbounds x[1])
    y = ntuple(i -> Int(@inbounds x[i + 1]), Val(M))
    return (g, _symcol(X, _tsort(y)))
end

# ---------------------------------------------------------------------
# indexing:  X[x]  with x a Tuple/Vector,  or  X[g, y...]
# ---------------------------------------------------------------------
@inline function Base.getindex(X::StateArray,
                               x::Union{Tuple,AbstractVector{<:Integer}})
    i, j = stateindex(X, x)
    return @inbounds X.data[i, j]
end

@inline function Base.setindex!(X::StateArray, v,
                                x::Union{Tuple,AbstractVector{<:Integer}})
    i, j = stateindex(X, x)
    @inbounds X.data[i, j] = v
    return v
end

# varargs form: X[g, y₁, …, y_{n-1}]
@inline Base.getindex(X::StateArray{T,N}, x::Vararg{Integer,N}) where {T,N} =
    X[x]
@inline Base.setindex!(X::StateArray{T,N}, v, x::Vararg{Integer,N}) where {T,N} =
    (X[x] = v)

# callable form: X(g, y...) / X(x)
@inline (X::StateArray{T,N})(x::Vararg{Integer,N}) where {T,N} = X[x]
@inline (X::StateArray)(x::Union{Tuple,AbstractVector{<:Integer}}) = X[x]

end # module