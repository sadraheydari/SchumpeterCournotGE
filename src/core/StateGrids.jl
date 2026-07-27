"""
    StateGrids

Grids and multidimensional interpolation for `StateArray`s whose states are
permutation-symmetric in all but the first component.

# What this provides

  * **Spacings** — how nodes are distributed inside `[lo, hi]`. Built in:
    [`LinearSpacing`](@ref), [`LogSpacing`](@ref),
    [`ShiftedLogSpacing`](@ref), [`PowerSpacing`](@ref), and
    [`WarpSpacing`](@ref) for an arbitrary user map `[0,1] → [0,1]`.
  * **[`Grid1D`](@ref)** — nodes along one axis, plus `locate`.
  * **[`StateGrid`](@ref)** — the pair of axes used by a state: one grid for
    `x[1]`, one for the symmetric part `x[2:end]`. Independent of `n`, so a
    single `StateGrid` serves value functions, policies, and any other array
    over the same discretisation.
  * **[`interp`](@ref) / [`Interpolant`](@ref)** — piecewise-linear
    interpolation with affine extrapolation, at real-valued states.

# The interpolation scheme

Kuhn (simplicial) interpolation. Given a state `x ∈ ℝⁿ`:

 1. locate `x[1]` in the `x`-grid and each `x[j]`, `j ≥ 2`, in the `y`-grid,
    giving a cell index `cⱼ` and a fractional coordinate `tⱼ` per axis;
 2. sort the `tⱼ` in decreasing order;
 3. walk from the low corner `c` to the high corner `c + 1`, adding one unit
    vector at a time in that order. The `n+1` corners visited are the
    vertices of the simplex containing `x`, and the weights are the
    successive gaps between the sorted `t`'s.

This costs `n+1` array lookups against `2ⁿ` for multilinear interpolation —
at `n = 5`, six lookups instead of thirty-two. The saving matters more here
than usual, because each lookup is a sort plus a rank, not a plain load.

**The interpolant is exactly permutation-symmetric.** Permuting `x[2:end]`
permutes the `cⱼ` and `tⱼ` together, so the sorted `t` sequence is unchanged
and the visited corners are the permuted images of the originals. Since
`StateArray` canonicalises every lookup, the value is bit-identical rather
than merely close — exactly so whenever the fractional coordinates are
distinct, which is the generic case. When two coordinates tie at the same
fractional position in *different* cells the two orderings are still equal
algebraically, differing only in the order of floating-point accumulation.
Ties carry zero weight, which is also why the scheme is continuous across
simplex faces.

# Extrapolation

Outside the grid the *cell* is clamped to the boundary cell but the
fractional coordinate is **not**: `t` is allowed to go below 0 or above 1.
The weights still sum to one, so the result is the unique affine
continuation of the boundary simplex — the function keeps moving with the
boundary gradient instead of flattening.

Properties: continuous with the interior interpolant; exact for affine
functions everywhere, inside and out; no extra cost; nothing truncated.

Be aware of what linear extrapolation does to a *concave* value function: it
continues the boundary slope where the truth is flattening, so it overstates
the value above the grid, and the error compounds when several coordinates
are out of range at once. Size the box to contain the ergodic set and treat
extrapolation as a safety net for the solver's excursions.

# Example

```julia
G = StateGrid(0.0, 50.0, 100, 50)                     # k_x = 100, k_y = 50
G = StateGrid(0.1, 50.0, 100, 50; spacing = LogSpacing())
G = StateGrid(0.0, 50.0, 100, 50; spacing = WarpSpacing(u -> u^1.5))

V = statearray(G, 4)              # 4-component states on this discretisation
V[7, 3, 9, 3] = 1.5

Ṽ = Interpolant(V, G)
Ṽ(2.4, 8.1, 0.3, 8.1)             # real-valued state
Ṽ([2.4, 0.3, 8.1, 8.1])           # same value: symmetric in x[2:end]
Ṽ(2.4, 8.1, 0.3, 999.0)           # extrapolated, not clipped
```
"""
module StateGrids

using ..SymStateArrays

export AbstractSpacing, LinearSpacing, LogSpacing, ShiftedLogSpacing,
       PowerSpacing, WarpSpacing, warp, gridnodes,
       Grid1D, StateGrid, nodes, locate, xaxis, yaxis,
       statearray, gridpoint, interp, Interpolant

# =====================================================================
#  Spacings
#
#  A spacing decides where the k nodes sit inside [lo, hi]. The single
#  method a spacing must provide is `gridnodes(s, lo, hi, k)`. The default
#  implementation warps a uniform mesh on [0,1] through `warp(s, u)`, so
#  most spacings need only that one scalar function; the log spacings
#  override `gridnodes` instead because they depend on lo and hi, not just
#  on the unit interval.
# =====================================================================

"""
    AbstractSpacing

Supertype for node placements. Implement either

    warp(s::MySpacing, u::Float64) -> Float64    # [0,1] → [0,1], increasing

for a placement expressible as a warp of the unit interval, or

    gridnodes(s::MySpacing, lo, hi, k) -> Vector{Float64}

when the placement depends on `lo` and `hi` themselves.
"""
abstract type AbstractSpacing end

"""
    warp(spacing, u) -> Float64

Map `u ∈ [0,1]` to `[0,1]`, strictly increasing with `warp(s,0) == 0` and
`warp(s,1) == 1`. Used by the default [`gridnodes`](@ref).
"""
function warp end

"""
    gridnodes(spacing, lo, hi, k) -> Vector{Float64}

The `k` nodes of `spacing` on `[lo, hi]`. The default implementation is
`lo + (hi-lo) * warp(spacing, u)` on a uniform `u`-mesh.
"""
function gridnodes(s::AbstractSpacing, lo::Real, hi::Real, k::Integer)
    lo, hi = Float64(lo), Float64(hi)
    return [lo + (hi - lo) * warp(s, (i - 1) / (k - 1)) for i in 1:k]
end

"""
    LinearSpacing()

Evenly spaced nodes. [`locate`](@ref) has an `O(1)` fast path for this
spacing rather than the usual binary search.
"""
struct LinearSpacing <: AbstractSpacing end
warp(::LinearSpacing, u::Float64) = u

"""
    PowerSpacing(θ)

Nodes at `lo + (hi-lo)·uᶿ`. `θ > 1` bunches nodes near `lo`, `θ < 1` near
`hi`. A common choice for asset grids is `θ = 2`, and unlike
[`LogSpacing`](@ref) it is happy with `lo == 0`.
"""
struct PowerSpacing <: AbstractSpacing
    θ::Float64
    function PowerSpacing(θ::Real)
        θ > 0 || throw(ArgumentError("θ must be positive (got $θ)"))
        new(Float64(θ))
    end
end
warp(s::PowerSpacing, u::Float64) = u^s.θ

"""
    WarpSpacing(f)

Arbitrary spacing from a user function `f: [0,1] → [0,1]`, which must be
strictly increasing with `f(0) == 0` and `f(1) == 1`. The grid constructor
checks monotonicity and pins the endpoints, so small numerical slop in `f`
is tolerated.

```julia
WarpSpacing(u -> u^1.5)
WarpSpacing(u -> (exp(3u) - 1) / (exp(3) - 1))   # exponential bunching
```
"""
struct WarpSpacing{F} <: AbstractSpacing
    f::F
end
warp(s::WarpSpacing, u::Float64) = Float64(s.f(u))

"""
    LogSpacing()

Geometric spacing: nodes evenly spaced in `log`, i.e. constant ratio
between neighbours. Requires `lo > 0` — use [`ShiftedLogSpacing`](@ref) or
[`PowerSpacing`](@ref) if the axis starts at zero.
"""
struct LogSpacing <: AbstractSpacing end

function gridnodes(::LogSpacing, lo::Real, hi::Real, k::Integer)
    lo > 0 || throw(ArgumentError(
        "LogSpacing needs lo > 0 (got $lo); try ShiftedLogSpacing or PowerSpacing"))
    return exp.(range(log(Float64(lo)), log(Float64(hi)); length = k))
end

"""
    ShiftedLogSpacing(c)

Geometric spacing in the shifted variable `v + c`, so the axis may start at
`lo == 0` provided `lo + c > 0`. Larger `c` moves the placement towards
uniform; smaller `c` bunches nodes harder near `lo`.
"""
struct ShiftedLogSpacing <: AbstractSpacing
    c::Float64
    ShiftedLogSpacing(c::Real) = new(Float64(c))
end

function gridnodes(s::ShiftedLogSpacing, lo::Real, hi::Real, k::Integer)
    lo + s.c > 0 || throw(ArgumentError(
        "ShiftedLogSpacing needs lo + c > 0 (got lo=$lo, c=$(s.c))"))
    return exp.(range(log(Float64(lo) + s.c), log(Float64(hi) + s.c);
                      length = k)) .- s.c
end

# =====================================================================
#  One axis
# =====================================================================

"""
    Grid1D(lo, hi, k; spacing = LinearSpacing())

`k` strictly increasing nodes on `[lo, hi]`, placed by `spacing`. The
endpoints are pinned to `lo` and `hi` exactly, absorbing any rounding in
the spacing map.

`nodes(g)` returns the node vector, `g[i]` the `i`-th node, `length(g)` the
count. Requires `k ≥ 2`: a single node admits no cell and so no
interpolation.
"""
struct Grid1D{S<:AbstractSpacing}
    lo::Float64
    hi::Float64
    pts::Vector{Float64}
    spacing::S
end

function Grid1D(lo::Real, hi::Real, k::Integer; spacing::AbstractSpacing = LinearSpacing())
    k >= 2 || throw(ArgumentError("a grid needs at least 2 nodes (got $k)"))
    lo < hi || throw(ArgumentError("need lo < hi (got lo=$lo, hi=$hi)"))
    p = collect(Float64, gridnodes(spacing, lo, hi, Int(k)))
    length(p) == k || throw(ArgumentError(
        "gridnodes returned $(length(p)) nodes, expected $k"))
    p[1], p[end] = Float64(lo), Float64(hi)      # pin the endpoints
    for i in 1:(k - 1)
        p[i] < p[i + 1] || throw(ArgumentError(
            "spacing must be strictly increasing; nodes $i and $(i+1) are " *
            "$(p[i]) and $(p[i+1])"))
    end
    return Grid1D{typeof(spacing)}(Float64(lo), Float64(hi), p, spacing)
end

"""
    nodes(g::Grid1D) -> Vector{Float64}

The node vector, in increasing order.
"""
nodes(g::Grid1D) = g.pts

Base.length(g::Grid1D)              = length(g.pts)
Base.getindex(g::Grid1D, i::Integer) = g.pts[i]
Base.firstindex(::Grid1D)           = 1
Base.lastindex(g::Grid1D)           = length(g.pts)
Base.extrema(g::Grid1D)             = (g.lo, g.hi)

function Base.show(io::IO, g::Grid1D)
    print(io, "Grid1D(", g.lo, ", ", g.hi, ", ", length(g.pts),
              "; spacing = ", g.spacing, ")")
end

"""
    locate(g::Grid1D, v::Real) -> (i, t)

Cell index `i ∈ 1:length(g)-1` and fractional coordinate
`t = (v - g[i]) / (g[i+1] - g[i])`.

For `v` inside the grid this is the containing cell and `t ∈ [0,1]`. For `v`
outside, **the cell is clamped to the boundary cell but `t` is not clamped**:
it comes back negative below the grid and greater than one above. That is
what makes extrapolation affine rather than flat — see the module docstring.

`t` is computed from the stored nodes, so it is exactly `0` at a node
regardless of spacing.
"""
@inline function locate(g::Grid1D, v::Real)
    p = g.pts
    k = length(p)
    i = searchsortedlast(p, v)
    i = i < 1 ? 1 : (i > k - 1 ? k - 1 : i)
    @inbounds t = (v - p[i]) / (p[i + 1] - p[i])
    return (i, t)
end

# Uniform nodes invert in closed form, so skip the binary search. The
# guards keep `floor` away from values that would not fit an Int.
@inline function locate(g::Grid1D{LinearSpacing}, v::Real)
    p = g.pts
    k = length(p)
    local i::Int
    if v <= g.lo
        i = 1
    elseif v >= g.hi
        i = k - 1
    else
        Δ = (g.hi - g.lo) / (k - 1)
        i = floor(Int, (v - g.lo) / Δ) + 1
        i = i < 1 ? 1 : (i > k - 1 ? k - 1 : i)
    end
    @inbounds t = (v - p[i]) / (p[i + 1] - p[i])
    return (i, t)
end

# =====================================================================
#  The state grid
# =====================================================================

"""
    StateGrid(gmin, gmax, kx, ky; spacing = LinearSpacing(), kwargs...)
    StateGrid(gmin, gmax, k; spacing = LinearSpacing(), kwargs...)      # k_x = k_y = k
    StateGrid(xaxis::Grid1D, yaxis::Grid1D)

The two axes a state lives on: `kx` nodes for `x[1]` and `ky` nodes for each
of the permutation-symmetric components `x[2:end]`.

Keyword arguments:

  * `spacing`  — node placement for the `x` axis, and for `y` unless
    `yspacing` is given;
  * `yspacing` — node placement for the `y` axis (defaults to `spacing`);
  * `ymin`, `ymax` — range of the `y` axis (defaults to `gmin`, `gmax`).

A `StateGrid` carries no notion of `n`, so one grid serves every array over
the same discretisation — value function, policies, whatever else.

```julia
G = StateGrid(0.0, 50.0, 100, 50)
G = StateGrid(0.1, 50.0, 100, 50; spacing = LogSpacing())
G = StateGrid(0.0, 50.0, 100, 50; spacing = PowerSpacing(2),
                                  yspacing = LinearSpacing())
```
"""
struct StateGrid{Sx<:AbstractSpacing,Sy<:AbstractSpacing}
    x::Grid1D{Sx}
    y::Grid1D{Sy}
end

function StateGrid(gmin::Real, gmax::Real, kx::Integer, ky::Integer;
                   spacing::AbstractSpacing = LinearSpacing(),
                   yspacing::AbstractSpacing = spacing,
                   ymin::Real = gmin, ymax::Real = gmax)
    return StateGrid(Grid1D(gmin, gmax, kx; spacing = spacing),
                     Grid1D(ymin, ymax, ky; spacing = yspacing))
end
function StateGrid(gmin::Real, gmax::Real, k::Integer;
                   spacing::AbstractSpacing = LinearSpacing())
    return StateGrid(gmin, gmax, k, k; spacing = spacing)
end

"""
    xaxis(G) -> Grid1D
    yaxis(G) -> Grid1D

The axis for `x[1]` and the axis shared by `x[2:end]`.
"""
xaxis(G::StateGrid) = G.x
yaxis(G::StateGrid) = G.y

function Base.show(io::IO, G::StateGrid)
    print(io, "StateGrid(x = ", G.x, ", y = ", G.y, ")")
end

"""
    statearray(G, n, T = Float64) -> StateArray

A zero-initialised `StateArray` of `n`-component states sized to `G`, i.e.
with `length(xaxis(G))` and `length(yaxis(G))` grid points. The convenient
way to allocate a value function or a policy that is guaranteed compatible
with `G`.

```julia
V = statearray(G, 4)          # Float64 value function
P = statearray(G, 4, Int)     # integer policy on the same grid
```
"""
statearray(G::StateGrid, n::Integer, ::Type{T} = Float64) where {T} =
    StateArray{T}(n, length(G.x), length(G.y))

"""
    gridpoint(G, x) -> NTuple{N,Float64}

The real coordinates of the state whose *grid indices* are `x`. The inverse
of what [`interp`](@ref) consumes, and what you want when sweeping the state
space and evaluating a model at each node:

```julia
for j in axes(V.data, 2), i in axes(V.data, 1)
    # ... recover the state indices, then:
    g, y... = gridpoint(G, state_indices)
end
```
"""
@inline gridpoint(G::StateGrid, x::NTuple{N,Integer}) where {N} =
    ntuple(j -> j == 1 ? G.x[x[1]] : G.y[x[j]], Val(N))
gridpoint(G::StateGrid, x::AbstractVector{<:Integer}) =
    [j == 1 ? G.x[x[1]] : G.y[x[j]] for j in eachindex(x)]

"""
    compatible(V::StateArray, G::StateGrid) -> Bool

Whether `V` was built for `G`: matching numbers of nodes on both axes.
"""
compatible(V::StateArray, G::StateGrid) =
    gridsize(V) == length(G.x) && ygridsize(V) == length(G.y)

function _checkcompat(V::StateArray, G::StateGrid)
    compatible(V, G) || throw(DimensionMismatch(
        "array is $(gridsize(V))×$(ygridsize(V)) grid points but grid is " *
        "$(length(G.x))×$(length(G.y))"))
    return nothing
end

# =====================================================================
#  Kuhn simplicial interpolation
#
#  Sort the fractional coordinates t₁…t_d in decreasing order, σ. The
#  simplex containing the point has vertices
#
#      v₀ = c,  vᵢ = vᵢ₋₁ + e_{σ(i)},  v_d = c + 1
#
#  with barycentric weights w₀ = 1 - t_{σ(1)}, wᵢ = t_{σ(i)} - t_{σ(i+1)},
#  w_d = t_{σ(d)}. Rather than form the weights, we use the telescoped
#  identity
#
#      Σᵢ wᵢ f(vᵢ)  =  f(v₀) + Σᵢ t_{σ(i)} · ( f(vᵢ) - f(vᵢ₋₁) )
#
#  which is a clean recursion over the sorted tuple and touches each corner
#  once. It stays valid verbatim when some tⱼ lie outside [0,1] — the
#  weights then sum to one with some negative, giving the affine
#  continuation of the same simplex.
# =====================================================================

# Insertion sort of a tuple of (t, dimension) pairs, decreasing in t.
# Same unrolled, allocation-free construction as the multiset sort.
@inline _sortdesc(t::Tuple{})    = ()
@inline _sortdesc(t::Tuple{Any}) = t
@inline _sortdesc(t::Tuple)      = _insertdesc(t[1], _sortdesc(Base.tail(t)))

@inline _insertdesc(v, ::Tuple{}) = (v,)
@inline function _insertdesc(v, t::Tuple)
    v[1] >= t[1][1] ? (v, t...) : (t[1], _insertdesc(v, Base.tail(t))...)
end

# corner with coordinate j incremented, without allocating
@inline _bump(c::NTuple{D,Int}, j::Int) where {D} =
    ntuple(l -> ifelse(l == j, c[l] + 1, c[l]), Val(D))

# Walk the simplex, accumulating Σ t·Δf. `prev` is f at the current corner.
# The base case constrains `corner` exactly as the recursive method does:
# leaving it untyped would make the two methods ambiguous for an empty
# `ord`, since neither would be more specific than the other.
@inline _walk(V, corner::NTuple{D,Int}, prev, ::Tuple{}) where {D} = 0.0
@inline function _walk(V, corner::NTuple{D,Int}, prev, ord::Tuple) where {D}
    t, j = ord[1]
    nxt  = _bump(corner, j)
    fv   = @inbounds V[nxt]
    return t * (fv - prev) + _walk(V, nxt, fv, Base.tail(ord))
end

"""
    interp(V, G, x)

Value of `V` at the real-valued state `x`, by simplicial interpolation on
`G`, with affine extrapolation outside the grid.

`x` is a `Tuple` or `AbstractVector` of `n` reals — `x[1]` on the `x` axis,
`x[2:end]` on the `y` axis and symmetric among themselves. `V` must be a
`StateArray` sized to `G`; see [`statearray`](@ref).

Costs `n+1` lookups into `V`, allocates nothing, and is exact at grid points
and for affine functions of the coordinates (including outside the grid).
The result is symmetric in `x[2:end]` exactly rather than approximately —
see the module docstring for the one floating-point caveat.

Accumulation is in `Float64` unless the element type of `V` or the state
promotes to something richer, so an `Int`-valued policy array interpolates
to `Float64`.

```julia
interp(V, G, (2.4, 8.1, 0.3, 8.1))
interp(V, G, [2.4, 0.3, 8.1, 8.1])    # identical
```

Bounds and compatibility are checked; `@inbounds` removes those checks.
"""
@inline function interp(V::StateArray{T,N,M}, G::StateGrid, x) where {T,N,M}
    @boundscheck begin
        length(x) == N || throw(DimensionMismatch(
            "state has length $(length(x)), expected $N"))
        _checkcompat(V, G)
    end
    # locate on every axis: axis 1 is the x grid, the rest share the y grid
    ct = ntuple(j -> j == 1 ? locate(G.x, @inbounds(x[1])) :
                              locate(G.y, @inbounds(x[j])), Val(N))
    cells = ntuple(j -> ct[j][1], Val(N))
    ord   = _sortdesc(ntuple(j -> (ct[j][2], j), Val(N)))
    f0    = @inbounds V[cells]
    return f0 + _walk(V, cells, f0, ord)
end

"""
    Interpolant(V, G)

Callable wrapper pairing an array with its grid, so the interpolated
function can be passed around as a function of the state:

```julia
Ṽ = Interpolant(V, G)
Ṽ(2.4, 8.1, 0.3, 8.1)      # varargs
Ṽ((2.4, 8.1, 0.3, 8.1))    # tuple
Ṽ([2.4, 8.1, 0.3, 8.1])    # vector
```

Compatibility of `V` with `G` is checked once, at construction. The wrapper
is immutable but holds `V` by reference, so writes to `V` are seen by an
existing `Interpolant` — build it once and keep using it across a Bellman
iteration.
"""
struct Interpolant{N,A<:StateArray,G<:StateGrid}
    V::A
    grid::G
end

function Interpolant(V::StateArray{T,N,M}, G::StateGrid) where {T,N,M}
    _checkcompat(V, G)
    return Interpolant{N,typeof(V),typeof(G)}(V, G)
end

@inline (f::Interpolant)(x::Union{Tuple,AbstractVector{<:Real}}) =
    interp(f.V, f.grid, x)
@inline (f::Interpolant{N})(x::Vararg{Real,N}) where {N} =
    interp(f.V, f.grid, x)

Base.show(io::IO, f::Interpolant{N}) where {N} =
    print(io, "Interpolant(n=", N, ") over ", f.grid)

end # module