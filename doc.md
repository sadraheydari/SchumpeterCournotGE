# The Detrended Problem: Definition and Derivation

This document defines, and fully derives, the stationary (detrended) recursive
problem solved by each firm. It is self-contained: everything needed to
implement the firm's dynamic program is stated here, together with a brief
justification tying each step back to the level (undetrended) equations of the
model. Notation matches the rest of the implementation reference.

---

## 1. Why detrending is necessary

Along any equilibrium path with positive R&D, productivity $A_i$, the wage
$w_t$, and aggregate output $\mathcal Y_t$ all grow without bound — this is the
whole point of the innovation technology. Posed directly in levels, the firm's
Bellman equation has a state space and a value function that both grow every
period, so there is no fixed point to compute and no stationary policy
function to solve for.

Detrending re-expresses the *same* problem in variables that are stationary
along the balanced growth path (BGP), so that a single time-invariant Bellman
equation and a single time-invariant policy $\pi_r(\cdot)$ characterize the
equilibrium. This is possible because of two properties of the static
equilibrium, established next.

---

## 2. Scale properties of the static equilibrium

Consider a common rescaling of every firm's productivity in the economy,
$A_i \to \kappa A_i$ for all $i$, in all industries (i.e. rescaling the entire
cross-sectional distribution $\Phi$). Two classes of object behave differently:

**Homogeneous of degree 1 (they scale by $\kappa$ — "trending" objects):**
$A_i$, the industry harmonic mean $\tilde A(j)$, the wage $w$, aggregate output
$\mathcal Y$, industry output $y(j)$, firm output $y_i$, firm/industry
dividends $d_i,\,d(j)$, and the value function $V$.

*Why:* the harmonic mean is homogeneous of degree 1 by construction. The wage
equation aggregates $\tilde A(j)/m(j)$ across industries via a CES-type
aggregator with $m(j)$ unchanged under rescaling (see below), so $w$ inherits
degree 1. $\mathcal Y = w(1-L^r)/(1-\mathscr L)$ then inherits degree 1 from
$w$ since $L^r,\mathscr L$ are degree 0. Dividends and firm value inherit
degree 1 by the same chain, since discounting depends only on *growth rates*,
not on levels.

**Homogeneous of degree 0 (unchanged — "stationary" objects):**
the relative price $p(j)$, the markup $m(j)$, the number of active firms
$\tilde n(j)$ (and hence the identity of the active set $\mathcal N^*$), market
shares $s_i$, Lerner indices $\ell_i,\mathscr L$, and the labour shares
$L^p, L^r$.

*Why:* every one of these is a ratio of two productivity levels, or a share of
labour (which is exogenously fixed at 1 and never rescales). A common
rescaling of productivities cancels in every ratio.

This is the reason the model is tractable: **only two** aggregate objects
($w_t$ and $\mathcal Y_t$) carry a trend, and everything else is already
scale-free.

---

## 3. Closing the model: the choice $\bar A_t \equiv w_t$

The theoretical draft defines the catch-up ratio $\rho_i = A_i/\bar A_t$ with
$\bar A_t = \Xi(\Phi_t)$, where $\Xi$ is *left unspecified* — only required to
be homogeneous of degree 1 — with the promise that a specific choice would be
tied to the equilibrium objects once the Cournot equilibrium was in hand.

**This is where that choice is made:** we set

$$
\boxed{\bar A_t \equiv w_t.}
$$

This is a modeling assumption (fixing $\Xi$), not a derived identity. It is a
natural choice because $w_t$ is (a) homogeneous of degree 1 in $\Phi_t$, exactly
as required of $\Xi$, and (b) already the equilibrium-consistent aggregator of
industry productivities (via the wage equation), so no additional aggregator
needs to be introduced. Given this choice,

$$
\rho_i = \frac{A_i}{\bar A_t} = \frac{A_i}{w_t},
$$

which is exactly the normalized productivity defined next — i.e. the
catch-up/standing-on-shoulders term collapses onto the detrended state itself,
with no separate object to track.

---

## 4. Detrended variables — definitions

**State:**
$$
a_i \equiv \frac{A_i}{w_t}.
$$
Because $A_i$ is degree 1 and $w_t$ is degree 1, $a_i$ is degree 0: a genuinely
stationary state variable (unlike raw $A_i$). Consequently $\rho_i = a_i$.

**Flows (all degree-1 objects divided through by $w_t$):**
$$
\hat V \equiv \frac{V}{w_t}, \qquad
\hat d_i \equiv \frac{d_i}{w_t}, \qquad
\hat y(j) \equiv \frac{y(j)}{w_t}, \qquad
\hat y_i \equiv \frac{y_i}{w_t}, \qquad
\hat{\mathcal Y} \equiv \frac{\mathcal Y_t}{w_t}.
$$

**Growth rates:**
$$
1+g_t^w \equiv \frac{w_{t+1}}{w_t}, \qquad
1+g_t^y \equiv \frac{\mathcal Y_{t+1}}{\mathcal Y_t}.
$$
(Market clearing gives $\mathcal Y_t=\mathcal C_t$, so $g^y$ is equivalently
consumption growth — this is the object that enters the household's
stochastic discount factor.)

Objects that are *already* degree 0 are **not** detrended and are used exactly
as in levels: $p(j)$, $m(j)$, $\tilde n(j)$, $s_i$, $\ell_i$, $\mathscr L$,
$L^p$, $L^r$, and the production/research labour allocations $l_i^p, l_i^r$
themselves (labour is never rescaled).

---

## 5. Static equilibrium, in detrended form

Given a state vector $\mathbf a=(a_1,\ldots,a_n)$ and a candidate active set
$\mathcal K$ of size $k$:

$$
\tilde a(\mathcal K) = \left(\frac1k\sum_{i\in\mathcal K}\frac1{a_i}\right)^{-1},
\qquad
m(\mathcal K) = \frac{\mu k}{\mu k - 1}.
$$

**Active set.** A set $\mathcal N^*$ of size $\tilde n$ is the equilibrium
active set iff, writing $\tilde a\equiv\tilde a(\mathcal N^*)$,
$$
a_i \ge \frac{\mu\tilde n - 1}{\mu\tilde n}\,\tilde a \quad \forall i\in\mathcal N^*,
\qquad
a_{i'} < \frac{\mu(\tilde n+1)-1}{\mu(\tilde n+1)}\,\tilde a(\mathcal N^*\cup\{i'\}) \quad \forall i'\notin\mathcal N^*.
$$
This is self-referential ($\tilde n,\tilde a$ depend on $\mathcal N^*$, which is
what is being solved for). The model's existence/uniqueness result guarantees
$\mathcal N^*$ consists of the $\tilde n$ most productive firms for a unique
$\tilde n$, which gives a direct algorithm:

> **Solving for $\mathcal N^*$ given $\mathbf a$:**
> 1. Sort firms in descending order of $a_i$.
> 2. For $k=n,n-1,\ldots,1$, compute $\tilde a_k \equiv \tilde a(\text{top }k)$.
> 3. Let $k^*$ be the largest $k$ such that the $k$-th ranked firm satisfies
>    $a_{(k)} \ge \frac{\mu k-1}{\mu k}\tilde a_k$.
> 4. Set $\mathcal N^*=$ top $k^*$ firms, $\tilde n=k^*$, $\tilde a=\tilde a_{k^*}$.

**Price:** since $\tilde A(j) = w_t\,\tilde a$,
$$
p = \frac{m\,w_t}{\tilde A(j)} = \frac{m}{\tilde a}.
$$

**Market share** (for $i\in\mathcal N^*$; else $s_i=0$):
$$
s_i = \max\left\{0,\; \mu\left(1-\frac{\tilde a}{m\,a_i}\right)\right\}.
$$

**Lerner index.** From $\ell_i=(1-1/(p a_i))s_i$ and $1/(pa_i)=\tilde a/(m a_i)=1-s_i/\mu$:
$$
\ell_i = \left(1-\frac1{p\,a_i}\right)s_i = \frac{s_i^2}{\mu}.
$$
(This also reproduces the aggregate identity $\mathrm{HHI}(j)=\mu\,\mathscr L(j)$.)

**Output, labour, dividend.** With $\hat{\mathcal Y}=(1-L^r)/(1-\mathscr L)$:
$$
\hat y(j) = \hat{\mathcal Y}\,p^{-\mu}, \qquad
\hat y_i = s_i\,\hat y(j), \qquad
l_i^p = \frac{\hat y_i}{a_i} \;\;(\text{= actual labour, not detrended}).
$$
$$
\hat d_i = p\,\hat y_i - l_i^p - l_i^r
\;=\;
p^{1-\mu}\,\ell_i\left(\frac{1-L^r}{1-\mathscr L}\right) - l_i^r.
$$
The two expressions are algebraically equivalent (using $\ell_i = s_i - l_i^p/(\hat y(j)\,p)$); **the second is used in the implementation** since it avoids computing $\hat y(j)$ and $l_i^p$ explicitly.

---

## 6. Innovation probability, detrended

Using $\rho_i=a_i$ from §3:
$$
\eta(l_i^r) = 1-\exp\!\left(-\bar\eta\,(l_i^r)^\theta\,a_i^{-\varepsilon}\right).
$$

---

## 7. Transition law, detrended

In levels, $A_{i,t+1}=\gamma A_{i,t}$ on success and $A_{i,t}$ on failure.
Dividing by $w_{t+1}=w_t(1+g_t^w)$:

$$
a_i' = \frac{\gamma\,a_i}{1+g_t^w} \quad(\text{success}), \qquad
a_i' = \frac{a_i}{1+g_t^w} \quad(\text{failure}).
$$

The next-period Bellman state $x_i' = (a_i',\, \mathbf a_{-i}'^{\,s})$ is
constructed by applying this transition **independently to every firm in the
industry** (each firm's own innovation draw, using its own $l_i^r$), dividing
every resulting productivity by the same $(1+g_t^w)$, and then re-sorting the
rivals' vector to preserve the sorted-state convention.

---

## 8. The detrended Bellman equation

Start from the level Bellman equation and household SDF, $M_{t,t+1}=\beta(1+g_t^y)^{-\sigma}$:
$$
V_t = \max_{l_i^r\ge0}\Big\{ d_{i,t} + M_{t,t+1}\,\mathbb E\big[V_{t+1}\big]\Big\}.
$$
Substitute $V_t = w_t\hat V_t$ and $V_{t+1}=w_{t+1}\hat V_{t+1}=w_t(1+g_t^w)\hat V_{t+1}$, then divide through by $w_t$:

$$
\boxed{\;
\hat V(x_i) = \max_{l_i^r\ge0}\Big\{\hat d_i(x_i,l_i^r) + \beta(1+g^y)^{-\sigma}(1+g^w)\,\mathbb E\big[\hat V(x_i')\big]\Big\}
\;}
$$

The expectation is over the **joint** innovation outcomes of all $n$ firms in
the industry (own draw and every rival's draw), since $x_i'$ depends on the
whole rival vector, not just firm $i$'s own transition. Rivals' research
choices enter through the symmetric equilibrium policy, $l_{i'}^r=\pi_r(x_{i'})$.

---

## 9. What the firm takes as given

**Idiosyncratic state:** $x_i = (a_i,\, \mathbf a_{-i}^{\,s})$ — own normalized
productivity and the sorted vector of rivals'.

**Aggregate objects, held fixed during value-function iteration:**
$$
(g^w,\; g^y,\; \hat{\mathcal Y}),
$$
where $\hat{\mathcal Y}=(1-L^r)/(1-\mathscr L)$ and $L^r,\mathscr L$ are
cross-sectional averages (labour share to research; economy-wide Lerner index)
computed from the *stationary* distribution of firm states under $\pi_r$ — i.e.
they are simulation-implied moments, not primitives, and are only updated in
the outer (general-equilibrium) loop, never inside the Bellman recursion.

---

## 10. Balanced-growth-path consistency

Along the BGP the cross-sectional moments $L^r,\mathscr L$ are constant (the
detrended distribution $\tilde\Phi_t\equiv\Phi_t/w_t$ is stationary), so
$\hat{\mathcal Y}$ is constant. Since $\mathcal Y_t = w_t\hat{\mathcal Y}$, this
forces

$$
g^w = g^y \equiv g \quad \text{(a single common growth rate on the BGP).}
$$

During the outer fixed-point iteration (before convergence) $g^w$ and $g^y$
need not coincide; $g^w=g^y$ is a *diagnostic* that the algorithm has reached
the BGP, in addition to the usual tolerance checks on $(g^w,g^y,\hat{\mathcal Y})$
individually.

---

## 11. Numerical Strategy

The model is solved using a nested fixed-point algorithm over aggregate objects and the firm's policy function.

1.  Take as given the aggregate measures: $(g_w, g_y, \hat{y})$.
2.  Take as given the policy guess: $\pi_r^{\text{comp}}$.
3.  Solve the firm's dynamic problem (Value Function Iteration or Policy Function Iteration) using current aggregate guesses and $\pi_r^{\text{comp}}$ to find the updated policy, $\pi_r$.
4.  Check policy convergence:
    *   If $||\pi_r - \pi_r^{\text{comp}}||_{\infty} > \text{tol}_1$:
        *   Update: $\pi_r^{\text{comp}} = \lambda \pi_r + (1-\lambda) \pi_r^{\text{comp}}$
        *   Return to **Step 2**.
5.  Simulate the economy over a panel of industries using the converged policy $\pi_r$ and find implied aggregate measures: $g'_w, g'_y, \hat{y}'$.
6.  Check aggregate convergence:
    *   If $\max\{|g_w - g'_w|, |g_y - g'_y|, |\hat{y} - \hat{y}'|\} > \text{tol}_2$:
        *   Perform a damped update on $(g_w, g_y, \hat{y})$.
        *   Return to **Step 1**.
7.  End convergence loop.