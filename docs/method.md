# Method

## 1. Setting

A fit returns best-fit parameters $\hat\theta$ and their covariance $\Sigma_\theta$.
Predictions in bins (momentum, angle) are a function $N(\theta)$. The question is:
**how does the uncertainty on $N_i$ change if the correlations in $\Sigma_\theta$ change?**

A covariance alone does not give predictions: a *response* $N(\theta)$ (or its
Jacobian) is required. Here it is synthetic, with known ground truth (`response.py`).

## 2. Propagation

**Linear.** With $J_{ij} = \partial N_i/\partial\theta_j |_{\hat\theta}$ (central finite differences):

$$\Sigma_N = J\,\Sigma_\theta\,J^T,\qquad \sigma_i = \sqrt{(\Sigma_N)_{ii}}$$

Exact if $N$ is linear in $\theta$ over the range allowed by $\Sigma_\theta$; first-order otherwise.

**Toys.** Cholesky $\Sigma_\theta = LL^T$, draw $z\sim\mathcal N(0,I)$, set $\theta=\hat\theta+Lz$,
evaluate $N(\theta)$ for many draws and take the empirical covariance. Handles non-linearity;
statistical precision on each $\sigma_i$ is $\approx 1/\sqrt{2n}$. If $\Sigma_\theta$ is only
positive *semi*-definite, an eigendecomposition replaces Cholesky.

Both are run on every covariance; their agreement is the validation (figure 05).
The same random numbers $z$ are reused across scenarios so that differences are not MC noise.

## 3. Covariance modifications

Write $\Sigma = D R D$, $D=\mathrm{diag}(\sigma)$, $R$ the correlation matrix.

| Scenario | Operation | PSD? |
|---|---|---|
| interpolate($\alpha$) | $R_\alpha=(1-\alpha)I+\alpha R$ | always (convex mix of PSD matrices) |
| decorrelate_groups(A,B) | zero the $R_{AB}$ block | not guaranteed → eigenvalue clipping + diagonal rescale (`ensure_psd`) |
| inflate(G, k) | $\sigma_G\to k\,\sigma_G$, $R$ unchanged | always |

## 4. Metrics

* $\sigma_{mod,i}/\sigma_{ref,i}$ per bin (headline metric).
* Change of the bin-to-bin correlation matrix (Frobenius norm / n_bins).
* **Group decomposition** of each bin variance: $\mathrm{Var}(N_i)=\sum_g J_g\Sigma_{gg}J_g^T + \text{cross-group}$.
  The cross-group term is negative when groups are anti-correlated: it measures *cancellation*,
  which is exactly what decorrelating destroys.

## 5. Why decorrelating can decrease or increase the uncertainty

Sensitivities of one group to a bin usually have the same sign.
* Positive correlations **inside** a group add coherently → removing them *reduces* $\sigma$.
* Anti-correlation **between** groups with same-sign sensitivities cancels → removing it *increases* $\sigma$.

The sign of the effect depends on the sign of $J_i\cdot R\cdot J_j$, not on "correlation" in the abstract.

## 6. Known limits

* Synthetic response: conclusions about numbers (+50 % etc.) belong to the benchmark, not to a real experiment.
* Gaussian parameters; linear propagation is first order (quantified in figure 05).
* Real fits need a real Jacobian or toys from the fitter; `io.read_gundam_fit` only reads $\hat\theta$ and $\Sigma$ (experimental, untested here).

## 7. Surrogate extension (optional)

Train $\hat f(\theta)\approx N(\theta)/N(\hat\theta)-1$ on $n$ draws from $\mathcal N(\hat\theta,\Sigma_{ref})$, then
estimate $\sigma_i$ for any covariance by sampling $\theta$ from it and taking the spread of $\hat f$.
Error is measured against the exact toys on independent samples. A surrogate is only reliable where it was trained:
modified covariances that move $\theta$ outside the training cloud are a genuine extrapolation test.
