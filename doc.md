# Economic Model Specification

This document serves as the implementation reference for the model. It contains all equations, state variables, parameters, normalization, and numerical procedures required for coding the model. The objective is that implementation should require consulting only this document, without referring back to the paper.

---

# 1. Overview

The economy consists of a continuum of industries. Each industry contains `n` firms that compete in quantities (Cournot competition) while simultaneously investing in R&D.

The solution algorithm searches for a **Symmetric Markov Perfect Equilibrium (SMPE)** together with a **Balanced Growth Path (BGP)**.

The numerical algorithm consists of two nested fixed-point problems:

1. **Inner fixed point**
   - Solve the firm's dynamic programming problem.
   - Compute the equilibrium research policy.

2. **Outer fixed point**
   - Simulate the economy using the policy.
   - Update aggregate growth rates until the simulated economy is consistent with the assumed aggregate variables.

The model is solved entirely in **detrended variables**.

---

# 2. Parameters

| Symbol | Description |
|---------|-------------|
| `β` | Discount factor |
| `σ` | Inv. Elasticity of intertemporal substitution |
| `μ` | Elasticity of substitution across industries |
| `γ` | Innovation step size |
| `η̄` | Innovation intensity |
| `θ` | Curvature of innovation function |
| `ε` | Catch-up parameter |
| `n` | Number of firms per industry |

---

# 3. State Variables

## Firm productivity

The model is solved using normalized productivity

$$
a_i=\frac{A_i}{w}.
$$

where

- $A_i$ is the productivity level
- $w$ is the aggregate wage.

The Bellman equation is solved over these normalized states.

---

## Bellman state

The state of firm $i$ is

$$
x_i=(a_i,\mathbf a_{-i}^{\,s}),
$$

where

- $a_i$ is own productivity,
- $\mathbf a_{-i}^{\,s}$ is the sorted productivity vector of competitors.

Sorting removes redundant permutations from the state space.

---

# 4. Aggregate Variables

The individual firm takes the following aggregate variables as given

$$
(g_w,\;g_y,\; \hat{\mathcal{Y}}),
$$

where

| Variable | Meaning |
|----------|----------|
| $g_w$ | Wage growth |
| $g_y$ | Aggregate output growth |
| $\hat{\mathcal{Y}} = \mathcal{Y}/w$ | Normalized aggregate output |

These variables are updated only in the outer fixed-point iteration.

---

# 5. Normalization

The model is solved using detrended variables.

The normalization is

$$
a_i=\frac{A_i}{w}.
$$

Furthermore,

$$
\boxed{\bar A=w.}
$$

Therefore

$$
\rho=\frac{A_i}{\bar A}
      =
      \frac{A_i}{w}
      =
      a_i.
$$

Consequently the innovation probability depends only on normalized productivity.

---

# 6. Innovation

The innovation probability is

$$
\eta(l)
=
1-
\exp\left(
-\bar\eta
l^\theta
\rho^{-\varepsilon}
\right).
$$

Using the normalization,

$$
\rho=a_i.
$$

Therefore

$$
\eta(l)
=
1-
\exp\left(
-\bar\eta
l^\theta
a_i^{-\varepsilon}
\right).
$$

---

# 7. Static Equilibrium

Given a productivity vector

$$
\mathbf a=(a_1,\ldots,a_n),
$$

the static Cournot problem is solved. The static equilibrium returns
- active firms,
- participation set,
- aggregate productivity,
- industry markup,
- equilibrium prices,
- market shares,
- Lerner indices,
- firm profits.

These are deterministic functions of the current productivity state.

---

For a candidate active set
$
\mathcal N^*,
$
the aggregate productivity is the harmonic mean
$$
\tilde a
=
\left(
\frac1{\tilde n}
\sum_{i\in\mathcal N^*}
\frac1{a_i}
\right)^{-1},
$$
where
$
\tilde n =
|\mathcal N^*|.
$

The Cournot markup is
$$
m
=
\frac{\mu\tilde n}
{\mu\tilde n-1}.
$$

The detrended industry price is

$$
p
=
\frac{m}{\tilde a}.
$$

Since
$
a_i=\frac{A_i}{w},
$
the equilibrium price is already normalized by the wage.


For every firm

$$
s_i
= \max \bigg\{ 0,
\mu
\left(
1-
\frac{\tilde a}
{m a_i}
\right)
\bigg\}.
$$

Firm-level Lerner index

$$
\ell_i
=
\left(
1-
\frac1{p a_i}
\right)
s_i
=
\frac{s_i^2}{\mu}.
$$

Normalized output satisfies

$$
\hat{\mathcal{Y}}
=
\frac{1-L^r}
{1-\mathscr L}.
$$


Industry output is

$$
y
=
\hat{\mathcal{Y}}
p^{-\mu}.
$$

Firm output

$$
y_i
=
s_i y.
$$

Production labour is

$$
l_i^p
=
\frac{y_i}{a_i}.
$$

Firm detrended dividend is

$$
\hat d_i
=
p\,y_i
-
l_i^p
-
l_i^r.
$$

Equivalently,

$$
\hat d_i
=
p^{1-\mu}
\ell_i
\left(
\frac{1-L^r}
{1-\mathscr L}
\right)
-
l_i^r.
$$

The second expression is used in the implementation.

---

# Dynamic Problem

The Bellman equation is

$$
\hat V(x_i)
=
\max_{l_i^r}
\left\{
\hat d_i
+
\beta
(1+g_y)^{-\sigma}
(1+g_w)
E[\hat V(x_i')]
\right\}.
$$

---

# Innovation

Innovation probability

$$
\eta(l_i^r)
=
1-
\exp
\left(
-\bar\eta
(l_i^r)^\theta
a_i^{-\varepsilon}
\right).
$$

---

# Transition Law

For each firm

Successful innovation

$$
a_i'
=
\frac{\gamma a_i}
{1+g_w},
$$

Failure

$$
a_i'
=
\frac{a_i}
{1+g_w}.
$$

The next-period productivity vector is obtained by applying this transition independently to every firm and sorting competitors.

---

# Active Set

A firm is active if

$$
a_i
\ge
\frac{\mu\tilde n-1}
{\mu\tilde n}
\tilde a.
$$

Otherwise

$$
y_i=s_i=l_i^p=\ell_i=0.
$$

---

# 8. Firm Payoff

The detrended dividend is

$$
\hat d_i
=
p^{1-\mu}
\ell_i
\left(
\frac{1-L^r}
{1-\mathscr L}
\right)
-
l_i^r.
$$

Everything except research labour is determined by the static equilibrium.

---

# 9. Bellman Equation

The detrended Bellman equation is

$$
\hat V(x_i)
=
\max_{l_i^r}
\left\{
\hat d_i
+
\beta
(1+g_y)^{-\sigma}
(1+g_w)
E
\left[
\hat V(x_i')
\right]
\right\}.
$$

During value-function iteration

- $g_w$,
- $g_y$,

are treated as fixed.

---

# 10. State Transition

Each firm's innovation is independent.

If innovation succeeds,

$$
a_i'
=
\frac{\gamma a_i}{1+g_w}.
$$

Otherwise,

$$
a_i'
=
\frac{a_i}{1+g_w}.
$$

The next-period state is generated by

1. drawing innovation outcomes,
2. updating every firm's productivity,
3. dividing by wage growth,
4. sorting competitors.

---

# 11. Policy Function

The equilibrium policy is

$$
\pi(x_i)=l_i^r.
$$

Implementation stores

```julia
policy[state]
```

which returns optimal research labour.

---

# 12. Numerical Strategy

## Inputs

Initial guesses

- wage growth $g_w$,
- output growth $g_y$,
- normalized output $\hat Y$,
- policy function.

## Procedure

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



# 15. Implementation Notes

- The entire dynamic problem is solved in detrended variables.
- Productivity states are normalized by the aggregate wage.
- The normalization $\bar A=w$ implies that the catch-up term depends only on normalized productivity.
- Static Cournot equilibrium is solved before evaluating continuation values.
- Aggregate variables remain fixed during Bellman iteration.
- Aggregate variables are updated only after simulation.
- The algorithm therefore consists of two nested fixed-point iterations:
    1. Policy iteration.
    2. General equilibrium iteration.