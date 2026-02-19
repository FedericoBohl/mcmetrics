# src/mcmetrics/results.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

import numpy as np
import scipy.stats as st
import torch


def _as_batched_xy(X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Enforce shapes:
      - X: (R, n, k)
      - y: (R, n)

    Allow single-sample input:
      - X: (n, k)
      - y: (n,)
    """
    if X.ndim == 2:
        X = X.unsqueeze(0)
    if y.ndim == 1:
        y = y.unsqueeze(0)

    if X.ndim != 3:
        raise ValueError(f"X must be 3D (R,n,k) or 2D (n,k). Got {tuple(X.shape)}")
    if y.ndim != 2:
        raise ValueError(f"y must be 2D (R,n) or 1D (n,). Got {tuple(y.shape)}")
    if X.shape[0] != y.shape[0] or X.shape[1] != y.shape[1]:
        raise ValueError(f"Batch/obs dims mismatch: X {tuple(X.shape)}, y {tuple(y.shape)}")

    return X, y


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    """Detach tensor and move to CPU NumPy array (required for SciPy)."""
    return x.detach().cpu().numpy()


def _from_numpy(x: np.ndarray, ref: torch.Tensor) -> torch.Tensor:
    """Create a tensor on the same device and dtype as `ref`."""
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)


def _fmt(x: float, digits: int = 3) -> str:
    """Format a float with fixed decimals."""
    return f"{x:.{digits}f}"


def _coerce_beta_true(beta_true: torch.Tensor, ref: torch.Tensor, k: int) -> torch.Tensor:
    """
    Coerce beta_true to shape (k,) on the same device/dtype as ref.
    Accepts shapes: (k,), (1,k), (k,1). Rejects anything else.
    """
    bt = beta_true
    if not isinstance(bt, torch.Tensor):
        bt = torch.as_tensor(bt)

    bt = bt.to(device=ref.device, dtype=ref.dtype)

    if bt.ndim == 1 and bt.shape[0] == k:
        return bt
    if bt.ndim == 2 and bt.shape == (1, k):
        return bt.squeeze(0)
    if bt.ndim == 2 and bt.shape == (k, 1):
        return bt.squeeze(1)

    raise ValueError(f"beta_true must have shape (k,), (1,k), or (k,1). Got {tuple(bt.shape)}")


@dataclass(frozen=True)
class OLSResults:
    """
    Results container designed for Monte Carlo.
    Memory-light friendly: fitted/resid/y can be None if not stored.
    """

    # Always stored
    params: torch.Tensor         # (R,k)
    vcov: torch.Tensor           # (R,k,k)
    sigma2: torch.Tensor         # (R,)
    ssr: torch.Tensor            # (R,) always stored (even if resid not stored)
    _nobs: int
    df_resid: int
    df_model: int
    has_const: bool = True

    # Optionally stored (to save memory)
    fitted: Optional[torch.Tensor] = None  # (R,n)
    resid: Optional[torch.Tensor] = None   # (R,n)
    y: Optional[torch.Tensor] = None       # (R,n)
    
    # Metadata
    model_name: str = "OLS"
    method_name: str = "Least Squares"
    cov_type: str = "nonrobust"  # "nonrobust", "HC0", "HC1"
    backend: str = "torch"
    param_names: Optional[list[str]] = None

    # Inference choice
    use_t: bool = True           # classic -> True, robust (default) -> False

    # Monte Carlo truth (optional)
    beta_true: Optional[torch.Tensor] = None  # expected shape (k,) or broadcastable variants

    @property
    def nobs(self) -> int:
        return int(self._nobs)

    @property
    def R(self) -> int:
        return int(self.params.shape[0])

    @property
    def k(self) -> int:
        return int(self.params.shape[1])

    @property
    def stderr(self) -> torch.Tensor:
        return torch.sqrt(torch.diagonal(self.vcov, dim1=-2, dim2=-1))

    @property
    def statvalues(self) -> torch.Tensor:
        """t-stats or z-stats, depending on use_t."""
        return self.params / self.stderr

    def predict(
        self,
        X_new: torch.Tensor,
        *,
        params: str = "replications",
    ) -> torch.Tensor:
        """
        Predict y given new design matrix X_new.

        Supported shapes
        - X_new: (n_new, k)            -> treated as common design for all replications
        - X_new: (R, n_new, k)         -> per-replication design

        params option
        - "replications": use per-replication params (default). Output shape:
            * if X_new is (n_new,k)      -> (R, n_new)
            * if X_new is (R,n_new,k)    -> (R, n_new)
        - "mean": use mean(params) across replications. Output shape:
            * if X_new is (n_new,k)      -> (n_new,)
            * if X_new is (R,n_new,k)    -> (R, n_new)
        - "median": use median(params) across replications. Same output shapes as "mean".
        """
        Xn = X_new
        if not isinstance(Xn, torch.Tensor):
            Xn = torch.as_tensor(Xn)

        Xn = Xn.to(device=self.params.device, dtype=self.params.dtype)

        if Xn.ndim == 2:
            n_new, k = Xn.shape
            if k != self.k:
                raise ValueError(f"X_new has k={k} but model has k={self.k}")

            if params == "replications":
                # Expand common X_new to (R, n_new, k) and use per-rep params
                Xb = Xn.unsqueeze(0).expand(self.R, n_new, k)               # (R,n_new,k)
                yhat = (Xb @ self.params.unsqueeze(-1)).squeeze(-1)         # (R,n_new)
                return yhat

            if params == "mean":
                beta = self.params.mean(dim=0)                               # (k,)
                return (Xn @ beta)                                           # (n_new,)

            if params == "median":
                beta = self.params.median(dim=0).values                      # (k,)
                return (Xn @ beta)                                           # (n_new,)

            raise ValueError("params must be one of {'replications','mean','median'}")

        if Xn.ndim == 3:
            Rn, n_new, k = Xn.shape
            if k != self.k:
                raise ValueError(f"X_new has k={k} but model has k={self.k}")

            if params == "replications":
                if Rn != self.R:
                    raise ValueError(f"X_new has R={Rn} but results have R={self.R}")
                yhat = (Xn @ self.params.unsqueeze(-1)).squeeze(-1)          # (R,n_new)
                return yhat

            if params == "mean":
                beta = self.params.mean(dim=0)                                # (k,)
                yhat = (Xn @ beta.view(1, k, 1)).squeeze(-1)                  # (R,n_new)
                return yhat

            if params == "median":
                beta = self.params.median(dim=0).values                       # (k,)
                yhat = (Xn @ beta.view(1, k, 1)).squeeze(-1)                  # (R,n_new)
                return yhat

            raise ValueError("params must be one of {'replications','mean','median'}")

        raise ValueError(f"X_new must be 2D (n_new,k) or 3D (R,n_new,k). Got {tuple(Xn.shape)}")

    def get_fitted(
        self,
        X: torch.Tensor,
        *,
        params: str = "replications",
    ) -> torch.Tensor:
        """
        Compute fitted values on-the-fly given a design matrix X, without storing them.

        This is a thin wrapper around predict(), but named to emphasize "in-sample fitted"
        reconstruction when the user passes the original X.

        Supported shapes
        - X: (n, k)         -> common design for all replications
        - X: (R, n, k)      -> per-replication design

        params option
        - "replications": use per-replication params (default)
        - "mean": use mean(params) across replications
        - "median": use median(params) across replications
        """
        return self.predict(X, params=params)


    def get_resid(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        params: str = "replications",
    ) -> torch.Tensor:
        """
        Compute residuals on-the-fly given (X, y), without storing them.

        Supported shapes
        - X: (n, k)
        y: (n,) or (R, n)
        - X: (R, n, k)
        y: (R, n)

        params option
        - "replications": residuals per replication
        - "mean"/"median": residuals using aggregated parameters (useful for reporting)
        """
        # Coerce inputs to tensors on correct device/dtype
        Xn = X if isinstance(X, torch.Tensor) else torch.as_tensor(X)
        yn = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)

        Xn = Xn.to(device=self.params.device, dtype=self.params.dtype)
        yn = yn.to(device=self.params.device, dtype=self.params.dtype)

        # Compute fitted
        yhat = self.get_fitted(Xn, params=params)

        # Match shapes and compute residuals
        if Xn.ndim == 2:
            n, k = Xn.shape
            if k != self.k:
                raise ValueError(f"X has k={k} but model has k={self.k}")

            if params == "replications":
                # yhat: (R, n)
                if yn.ndim == 1:
                    if yn.shape[0] != n:
                        raise ValueError(f"y has n={yn.shape[0]} but X has n={n}")
                    yb = yn.view(1, n).expand(self.R, n)  # (R,n)
                    return yb - yhat
                if yn.ndim == 2:
                    if yn.shape != (self.R, n):
                        raise ValueError(f"y must be (R,n)={(self.R,n)}. Got {tuple(yn.shape)}")
                    return yn - yhat
                raise ValueError(f"y must be 1D (n,) or 2D (R,n). Got {tuple(yn.shape)}")

            # params in {"mean","median"}: yhat is (n,)
            if yn.ndim != 1:
                raise ValueError(f"When params='{params}', y must be 1D (n,). Got {tuple(yn.shape)}")
            if yn.shape[0] != n:
                raise ValueError(f"y has n={yn.shape[0]} but X has n={n}")
            return yn - yhat

        if Xn.ndim == 3:
            Rn, n, k = Xn.shape
            if k != self.k:
                raise ValueError(f"X has k={k} but model has k={self.k}")

            if yn.ndim != 2:
                raise ValueError(f"With X 3D, y must be 2D (R,n). Got {tuple(yn.shape)}")
            if yn.shape != (Rn, n):
                raise ValueError(f"y must match X batch/obs dims {(Rn,n)}. Got {tuple(yn.shape)}")

            # yhat shape:
            # - params="replications": (R,n) and requires Rn==self.R (enforced inside predict())
            # - params="mean"/"median": (Rn,n)
            return yn - yhat

        raise ValueError(f"X must be 2D (n,k) or 3D (R,n,k). Got {tuple(Xn.shape)}")

    def durbin_watson_on(
        self,
        X,
        y,
        *,
        params: str = "replications",
    ) -> torch.Tensor:
        """
        Durbin-Watson computed on-the-fly from (X,y) without storing residuals.

        Returns
        - if params="replications": (R,)
        - if params in {"mean","median"}: scalar tensor ()
        """
        e = self.get_resid(X, y, params=params)

        if e.ndim == 1:
            # single residual series (n,)
            de = e[1:] - e[:-1]
            num = (de * de).sum()
            den = (e * e).sum()
            return num / den

        if e.ndim == 2:
            # (R,n)
            de = e[:, 1:] - e[:, :-1]
            num = (de * de).sum(dim=1)
            den = (e * e).sum(dim=1)
            return num / den

        raise ValueError(f"Unexpected residual shape {tuple(e.shape)}")

    def _resid_for_plot(self, X=None, y=None) -> torch.Tensor:
        """
        Get residuals for plotting/diagnostics.
        - If self.resid is stored, uses it.
        - Else requires X and y to compute residuals on-the-fly.
        Returns (R,n).
        """
        if self.resid is not None:
            return self.resid
        if X is None or y is None:
            raise ValueError("Need X and y to compute residuals on-the-fly when resid is not stored.")
        return self.get_resid(X, y, params="replications")

    def qq_resid_on(
        self,
        X=None,
        y=None,
        *,
        r: int = 0,
        standardize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Normal Q-Q plot data for residuals of replication r.

        Returns
        - theo_q: theoretical Normal quantiles (n,)
        - samp_q: sample quantiles from residuals (n,)

        If standardize=True, residuals are demeaned and scaled by their sample std.
        """
        e = self._resid_for_plot(X=X, y=y)  # (R,n)
        R, n = e.shape
        if not (0 <= r < R):
            raise ValueError(f"r must be in [0,{R-1}]. Got {r}.")

        er = e[r].detach().cpu().numpy().astype(float)  # (n,)

        if standardize:
            m = float(er.mean())
            s = float(er.std(ddof=1)) if n > 1 else float(er.std())
            if s > 0:
                er = (er - m) / s
            else:
                er = er - m

        samp_q = np.sort(er)

        # Normal theoretical quantiles at plotting positions
        p = (np.arange(1, n + 1) - 0.5) / n
        theo_q = st.norm.ppf(p)

        return theo_q, samp_q

    def acf_resid_on(
        self,
        X=None,
        y=None,
        *,
        r: int = 0,
        nlags: int = 20,
        demean: bool = True,
    ) -> np.ndarray:
        """
        Compute residual ACF for replication r up to nlags.

        Returns
        - acf: (nlags+1,) with acf[0]=1

        ACF definition:
        rho_l = sum_{t=l..n-1} (e_t - m)(e_{t-l} - m) / sum_{t=0..n-1} (e_t - m)^2
        """
        e = self._resid_for_plot(X=X, y=y)  # (R,n)
        R, n = e.shape
        if not (0 <= r < R):
            raise ValueError(f"r must be in [0,{R-1}]. Got {r}.")

        nl = int(min(max(nlags, 0), n - 1))
        er = e[r].detach().cpu().numpy().astype(float)  # (n,)

        if demean:
            er = er - float(er.mean())

        denom = float(np.dot(er, er))
        if denom <= 0:
            out = np.zeros(nl + 1, dtype=float)
            out[0] = 1.0
            return out

        acf = np.empty(nl + 1, dtype=float)
        acf[0] = 1.0
        for lag in range(1, nl + 1):
            acf[lag] = float(np.dot(er[lag:], er[:-lag]) / denom)

        return acf

    def plot(
        self,
        *,
        kind: str = "params",
        j: int = 0,
        r: int = 0,
        X=None,
        y=None,
        bins: int = 30,
        nlags: int = 20,
        confint: bool = True,
        alpha: float = 0.05,
        show: bool = True,
        return_fig: bool = False,
        latex: bool = True,
        use_tex: bool = False,
    ):
        """
        Plot utilities (LaTeX-like style).

        kind options
        - "params"          : histogram of params[:, j]
        - "resid"           : residual series for replication r (needs stored resid or X,y)
        - "resid_hist"      : histogram of residuals for replication r (needs stored resid or X,y)
        - "fitted_vs_actual": scatter of y vs fitted for replication r (needs X,y)

        If latex=True, applies a LaTeX-like style. If use_tex=True, enables full LaTeX rendering
        (requires a LaTeX installation).
        """
        import matplotlib.pyplot as plt

        if latex:
            from mcmetrics.plotting import apply_latex_style
            apply_latex_style(use_tex=use_tex)

        # Resolve names
        names = self.param_names if self.param_names is not None else [f"beta[{i}]" for i in range(self.k)]
        if not (0 <= j < self.k):
            raise ValueError(f"j must be in [0,{self.k-1}]. Got {j}.")

        fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)

        if kind == "params":
            data = self.params[:, j].detach().cpu().numpy()
            ax.hist(data, bins=bins)
            ax.set_title(f"Monte Carlo distribution of {names[j]}")
            ax.set_xlabel(names[j])
            ax.set_ylabel("Frequency")

            # Optional: vertical line at beta_true
            if self.beta_true is not None:
                bt = _coerce_beta_true(self.beta_true, ref=self.params, k=self.k)
                ax.axvline(float(bt[j].item()), linestyle="--")

            # Optional: CI band for aggregated inference (informative)
            ci = self.conf_int(alpha=alpha, use_t=self.use_t)  # (R,k,2)
            lo = float(ci[:, j, 0].mean().item())
            hi = float(ci[:, j, 1].mean().item())
            ax.axvline(lo, linestyle=":")
            ax.axvline(hi, linestyle=":")

        elif kind in {"resid", "resid_hist"}:
            if self.resid is None:
                if X is None or y is None:
                    raise ValueError("Need X and y to compute residuals on-the-fly when resid is not stored.")
                e = self.get_resid(X, y, params="replications")
            else:
                e = self.resid

            if not (0 <= r < e.shape[0]):
                raise ValueError(f"r must be in [0,{e.shape[0]-1}]. Got {r}.")

            er = e[r].detach().cpu().numpy()

            if kind == "resid":
                ax.plot(er)
                ax.axhline(0.0, linestyle="--")
                ax.set_title(f"Residual series (replication r={r})")
                ax.set_xlabel("Observation")
                ax.set_ylabel("Residual")

            else:
                ax.hist(er, bins=bins)
                ax.axvline(0.0, linestyle="--")
                ax.set_title(f"Residual histogram (replication r={r})")
                ax.set_xlabel("Residual")
                ax.set_ylabel("Frequency")

        elif kind == "fitted_vs_actual":
            if X is None or y is None:
                raise ValueError("Need X and y for kind='fitted_vs_actual'.")
            yhat = self.get_fitted(X, params="replications")
            # coerce y into batched tensor on device
            yt = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
            yt = yt.to(device=self.params.device, dtype=self.params.dtype)
            if yt.ndim == 1:
                yt = yt.unsqueeze(0).expand(self.R, -1)

            if not (0 <= r < yhat.shape[0]):
                raise ValueError(f"r must be in [0,{yhat.shape[0]-1}]. Got {r}.")

            xr = yt[r].detach().cpu().numpy()
            yr = yhat[r].detach().cpu().numpy()
            ax.scatter(xr, yr)
            ax.set_title(f"Fitted vs actual (replication r={r})")
            ax.set_xlabel("Actual y")
            ax.set_ylabel("Fitted y")

        elif kind == "qq_resid":
            if X is None or y is None:
                # If resid stored, X/y not needed; otherwise required by helper
                pass
            theo_q, samp_q = self.qq_resid_on(X=X, y=y, r=r, standardize=True)
            ax.scatter(theo_q, samp_q)
            # 45-degree reference line
            lo = min(theo_q.min(), samp_q.min())
            hi = max(theo_q.max(), samp_q.max())
            ax.plot([lo, hi], [lo, hi], linestyle="--")
            ax.set_title(f"Normal Q-Q plot of residuals (replication r={r})")
            ax.set_xlabel("Theoretical quantiles")
            ax.set_ylabel("Sample quantiles")
            
        elif kind == "acf_resid":
            acf = self.acf_resid_on(X=X, y=y, r=r, nlags=nlags, demean=True)
            lags = np.arange(acf.shape[0])

            ax.bar(lags, acf)
            ax.axhline(0.0, linestyle="--")
            ax.set_title(f"Residual ACF (replication r={r})")
            ax.set_xlabel("Lag")
            ax.set_ylabel("ACF")

            if confint:
                n = float(self.nobs)
                band = 1.96 / np.sqrt(n) if n > 0 else 0.0
                ax.axhline(band, linestyle=":")
                ax.axhline(-band, linestyle=":")       
        else:
            raise ValueError("kind must be one of {'params','resid','resid_hist','fitted_vs_actual'}")

        if show:
            plt.show()

        if return_fig:
            return fig, ax

        if not show:
            plt.close(fig)

        return None

    def _pvalues_from_stat(self, stat: torch.Tensor, use_t: Optional[bool] = None) -> torch.Tensor:
        """
        Two-sided p-values for a given statistic tensor.

        - If use_t=True: Student-t(df_resid)
        - If use_t=False: Normal(0,1)
        """
        if use_t is None:
            use_t = self.use_t

        s = torch.abs(stat)
        s_np = _to_numpy(s)

        if use_t:
            p_np = 2.0 * st.t.sf(s_np, df=self.df_resid)
        else:
            p_np = 2.0 * st.norm.sf(s_np)

        return _from_numpy(p_np, ref=s)

    @property
    def pvalues(self) -> torch.Tensor:
        """
        Two-sided p-values computed via SciPy.

        - If use_t=True: Student-t(df_resid)
        - If use_t=False: Normal(0,1) (z-test)
        """
        return self._pvalues_from_stat(self.statvalues, use_t=self.use_t)

    @property
    def tss(self) -> Optional[torch.Tensor]:
        """Total sum of squares (R,). Requires y to be stored."""
        if self.y is None:
            return None
        if self.has_const:
            ybar = self.y.mean(dim=1, keepdim=True)
            return ((self.y - ybar) ** 2).sum(dim=1)
        return (self.y ** 2).sum(dim=1)

    @property
    def rsquared(self) -> Optional[torch.Tensor]:
        tss = self.tss
        if tss is None:
            return None
        return 1.0 - self.ssr / tss

    @property
    def adj_rsquared(self) -> Optional[torch.Tensor]:
        r2 = self.rsquared
        if r2 is None:
            return None
        n = float(self.nobs)
        if self.has_const:
            return 1.0 - (1.0 - r2) * (n - 1.0) / float(self.df_resid)
        return 1.0 - (1.0 - r2) * n / float(self.df_resid)

    @property
    def fvalue(self) -> Optional[torch.Tensor]:
        """
        Global F-test for slopes = 0 (excluding constant if has_const).
        Requires y to compute TSS (and hence ESS).
        """
        tss = self.tss
        if tss is None or self.df_model <= 0:
            return None
        ess = tss - self.ssr
        msr = ess / float(self.df_model)
        mse = self.ssr / float(self.df_resid)
        return msr / mse

    @property
    def f_pvalue(self) -> Optional[torch.Tensor]:
        f = self.fvalue
        if f is None:
            return None
        f_np = _to_numpy(f)
        p_np = st.f.sf(f_np, dfn=self.df_model, dfd=self.df_resid)
        return _from_numpy(p_np, ref=f)

    @property
    def llf(self) -> torch.Tensor:
        """
        Gaussian log-likelihood under homoskedastic normal errors.
        Uses sigma2_mle = SSR / n (statsmodels-style).
        Returns (R,).
        """
        n = float(self.nobs)
        sigma2_mle = self.ssr / n
        return -0.5 * n * (torch.log(2.0 * torch.pi * sigma2_mle) + 1.0)

    @property
    def aic(self) -> torch.Tensor:
        return -2.0 * self.llf + 2.0 * float(self.k)

    @property
    def bic(self) -> torch.Tensor:
        return -2.0 * self.llf + np.log(float(self.nobs)) * float(self.k)

    @property
    def durbin_watson(self) -> Optional[torch.Tensor]:
        """
        Durbin-Watson per replication (R,).
        Requires resid to be stored; otherwise returns None.
        """
        if self.resid is None:
            return None
        if self.nobs < 2:
            return None
        de = self.resid[:, 1:] - self.resid[:, :-1]
        return (de ** 2).sum(dim=1) / self.ssr

    def conf_int(self, alpha: float = 0.05, use_t: Optional[bool] = None) -> torch.Tensor:
        """
        Confidence intervals per replication.

        Output shape: (R, k, 2) where [:,:,0]=lower and [:,:,1]=upper.

        - If use_t is None, uses self.use_t.
        - If use_t is True -> t critical with df_resid.
        - Else -> z critical.
        """
        if use_t is None:
            use_t = self.use_t

        q = 1.0 - alpha / 2.0
        if use_t:
            crit = float(st.t.ppf(q, df=self.df_resid))
        else:
            crit = float(st.norm.ppf(q))

        crit_t = torch.as_tensor(crit, device=self.params.device, dtype=self.params.dtype)
        half = crit_t * self.stderr
        lo = self.params - half
        hi = self.params + half
        return torch.stack([lo, hi], dim=-1)

    def mc_bias(self) -> Optional[torch.Tensor]:
        """Monte Carlo bias: E[beta_hat] - beta_true, shape (k,)."""
        if self.beta_true is None:
            return None
        bt = _coerce_beta_true(self.beta_true, ref=self.params, k=self.k)
        return self.params.mean(dim=0) - bt

    def mc_rmse(self) -> Optional[torch.Tensor]:
        """Monte Carlo RMSE: sqrt(E[(beta_hat - beta_true)^2]), shape (k,)."""
        if self.beta_true is None:
            return None
        bt = _coerce_beta_true(self.beta_true, ref=self.params, k=self.k)
        err = self.params - bt.view(1, -1)
        return torch.sqrt((err * err).mean(dim=0))

    def coverage(self, alpha: float = 0.05, use_t: Optional[bool] = None) -> Optional[torch.Tensor]:
        """
        Coverage rate for beta_true using per-replication confidence intervals.
        Returns shape (k,) with coverage probabilities.
        """
        if self.beta_true is None:
            return None
        bt = _coerce_beta_true(self.beta_true, ref=self.params, k=self.k)
        ci = self.conf_int(alpha=alpha, use_t=use_t)
        lo = ci[:, :, 0]
        hi = ci[:, :, 1]
        inside = (bt.view(1, -1) >= lo) & (bt.view(1, -1) <= hi)
        return inside.to(self.params.dtype).mean(dim=0)

    # --------------------------
    # Size/Power: componentwise
    # --------------------------
    def stat_beta0(self, beta0: torch.Tensor, use_t: Optional[bool] = None) -> torch.Tensor:
        """
        Per-replication test statistic for H0: beta = beta0 (componentwise).
        Returns (R,k).
        """
        b0 = _coerce_beta_true(beta0, ref=self.params, k=self.k)
        return (self.params - b0.view(1, -1)) / self.stderr

    def pvalues_beta0(self, beta0: torch.Tensor, use_t: Optional[bool] = None) -> torch.Tensor:
        """Two-sided p-values for H0: beta = beta0 (componentwise). Returns (R,k)."""
        stat = self.stat_beta0(beta0, use_t=use_t)
        return self._pvalues_from_stat(stat, use_t=use_t)

    def reject_beta0(
        self,
        beta0: torch.Tensor,
        alpha: float = 0.05,
        use_t: Optional[bool] = None,
    ) -> torch.Tensor:
        """Rejection indicators for H0: beta = beta0. Returns (R,k) bool."""
        return self.pvalues_beta0(beta0, use_t=use_t) < alpha

    def rejection_rate_beta0(
        self,
        beta0: torch.Tensor,
        alpha: float = 0.05,
        use_t: Optional[bool] = None,
    ) -> torch.Tensor:
        """Rejection rates across replications for H0: beta = beta0. Returns (k,)."""
        return self.reject_beta0(beta0, alpha=alpha, use_t=use_t).to(self.params.dtype).mean(dim=0)

    # --------------------------
    # Wald test: R beta = q
    # --------------------------
    def wald_test(
        self,
        R_mat: torch.Tensor,
        q_vec: torch.Tensor,
        *,
        alpha: float = 0.05,
        use_f: Optional[bool] = None,
        use_t: Optional[bool] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Batched Wald test for linear restrictions: H0: R beta = q.

        Inputs
        - R_mat: (m,k)
        - q_vec: (m,)

        Returns dict with:
        - stat : (R,) Wald statistic (Chi^2_m) or F(m, df_resid)
        - pval : (R,) p-values
        - rej  : (R,) rejection indicators at alpha

        Conventions:
        - If use_t is None: defaults to self.use_t.
        - If use_f is None: use_f = use_t (classic) else Chi^2 asymptotic.
        """
        if use_t is None:
            use_t = self.use_t
        if use_f is None:
            use_f = bool(use_t)

        Rm = torch.as_tensor(R_mat, device=self.params.device, dtype=self.params.dtype)
        qv = torch.as_tensor(q_vec, device=self.params.device, dtype=self.params.dtype)

        if Rm.ndim != 2:
            raise ValueError(f"R_mat must be 2D (m,k). Got {tuple(Rm.shape)}")
        if qv.ndim != 1:
            raise ValueError(f"q_vec must be 1D (m,). Got {tuple(qv.shape)}")

        m, k = Rm.shape
        if k != self.k:
            raise ValueError(f"R_mat has k={k} but params have k={self.k}")
        if qv.shape[0] != m:
            raise ValueError(f"q_vec length {qv.shape[0]} must match m={m}")

        rbeta = torch.einsum("mk,rk->rm", Rm, self.params)
        diff = rbeta - qv.view(1, -1)

        RV = torch.einsum("mk,rkj->rmj", Rm, self.vcov)
        RVRT = torch.einsum("rmk,lk->rml", RV, Rm)

        sol = torch.linalg.solve(RVRT, diff.unsqueeze(-1))
        W = torch.einsum("rm,rm->r", diff, sol.squeeze(-1))

        W_np = _to_numpy(W)

        if use_f:
            F_np = W_np / float(m)
            p_np = st.f.sf(F_np, dfn=m, dfd=self.df_resid)
            stat = _from_numpy(F_np, ref=W)
        else:
            p_np = st.chi2.sf(W_np, df=m)
            stat = W

        pval = _from_numpy(p_np, ref=W)
        rej = pval < alpha
        return {"stat": stat, "pval": pval, "rej": rej}

    # --------------------------
    # Summary (MC-friendly)
    # --------------------------
    def summary(
        self,
        param_names: Optional[Sequence[str]] = None,
        digits: int = 4,
        alpha: float = 0.05,
        mc_unbiased: bool = True,
        beta_null: Optional[torch.Tensor] = None,
    ) -> str:
        """
        Statsmodels-like summary adapted to Monte Carlo.

        Coef table aggregates across replications:
          - coef    : mean(params)
          - MC sd   : std(params)
          - std err : mean(stderr)
          - stat    : mean(t or z)
          - P>|stat|: mean(pvalues)
          - [q_low, q_high] : empirical quantiles of params
          - Rej@alpha: rejection frequency for H0: beta=0

        If beta_true is provided:
          - Bias, RMSE, Coverage@{1-alpha} based on conf_int()

        If beta_null is provided:
          - Rejection rate for H0: beta = beta_null (componentwise)
            interpreted as size if beta_true == beta_null, otherwise power.
        """
        width = 100
        line = "=" * width
        dash = "-" * width

        if param_names is None:
            if self.param_names is not None:
                param_names = self.param_names
            else:
                param_names = [f"beta[{j}]" for j in range(self.k)]  
              
        now = datetime.now().strftime("%a, %d %b %Y  %H:%M:%S")

        def agg(x: Optional[torch.Tensor]) -> tuple[Optional[float], Optional[float]]:
            if x is None:
                return None, None
            mean = x.mean().item()
            med = x.median(dim=0).values.item()
            return float(mean), float(med)

        r2m, r2med = agg(self.rsquared)
        ar2m, ar2med = agg(self.adj_rsquared)
        fm, _ = agg(self.fvalue)
        fpm, _ = agg(self.f_pvalue)
        llm, llmed = agg(self.llf)
        aicm, aicmed = agg(self.aic)
        bicm, bicmed = agg(self.bic)
        dwm, dwmed = agg(self.durbin_watson)

        b_mean = self.params.mean(dim=0)
        b_mc_sd = self.params.std(dim=0, unbiased=mc_unbiased)
        se_mean = self.stderr.mean(dim=0)
        stat_mean = self.statvalues.mean(dim=0)
        p_mean = self.pvalues.mean(dim=0)

        q_low = float(alpha / 2.0)
        q_high = float(1.0 - alpha / 2.0)
        b_qlo = torch.quantile(self.params, q_low, dim=0)
        b_qhi = torch.quantile(self.params, q_high, dim=0)

        rej = (self.pvalues < alpha).to(self.params.dtype).mean(dim=0)

        stat_label = "t" if self.use_t else "z"
        p_label = "P>|t|" if self.use_t else "P>|z|"
        test_label = "t-test" if self.use_t else "z-test"

        title = f"{self.model_name} Monte Carlo Regression Results"
        out: list[str] = []
        out.append(title.center(width))
        out.append(line)

        col_left = width // 2
        col_right = width - col_left

        def pair(l: str, r: str) -> str:
            l = (l or "")[:col_left]
            r = (r or "")[:col_right]
            return f"{l:<{col_left}}{r:<{col_right}}"

        left: list[str] = []
        right: list[str] = []

        left.append(f"{'Model:':<25}{self.model_name}")
        right.append(f"{'Method:':<25}{self.method_name}")

        left.append(f"{'MC Replications:':<25}{self.R}")
        right.append(f"{'Backend:':<25}{self.backend}")

        left.append(f"{'No. Observations:':<25}{self.nobs}")
        right.append(f"{'Covariance Type:':<25}{self.cov_type}")

        left.append(f"{'Df Residuals:':<25}{self.df_resid}")
        right.append(f"{'Df Model:':<25}{self.df_model}")

        left.append(f"{'Date:':<25}{now[:16]}")
        right.append(f"{'Time:':<25}{now[-8:]}")

        if r2m is not None:
            left.append(f"{'R-squared (mean):':<25}{_fmt(r2m, digits)}")
            right.append(f"{'R-squared (median):':<25}{_fmt(r2med, digits)}")
        if ar2m is not None:
            left.append(f"{'Adj. R-squared:':<25}{_fmt(ar2m, digits)}")
            right.append(f"{'Adj. R^2 (med):':<25}{_fmt(ar2med, digits)}")
        if fm is not None:
            left.append(f"{'F-statistic (mean):':<25}{_fmt(fm, digits)}")
            right.append(f"{'Prob(F) (mean):':<25}{_fmt(fpm, digits) if fpm is not None else 'NA'}")
        if llm is not None:
            left.append(f"{'Log-Likelihood:':<25}{_fmt(llm, digits)}")
            right.append(f"{'LLF (median):':<25}{_fmt(llmed, digits)}")
        if aicm is not None:
            left.append(f"{'AIC:':<25}{_fmt(aicm, digits)}")
            right.append(f"{'AIC (median):':<25}{_fmt(aicmed, digits)}")
        if bicm is not None:
            left.append(f"{'BIC:':<25}{_fmt(bicm, digits)}")
            right.append(f"{'BIC (median):':<25}{_fmt(bicmed, digits)}")
        if dwm is not None:
            left.append(f"{'Durbin-Watson:':<25}{_fmt(dwm, digits)}")
            right.append(f"{'DW (median):':<25}{_fmt(dwmed, digits)}")

        for i in range(max(len(left), len(right))):
            out.append(pair(left[i] if i < len(left) else "", right[i] if i < len(right) else ""))

        out.append(line)
        out.append("Monte Carlo coefficient distribution (across replications)".center(width))
        out.append(dash)

        name_w = 14
        num_w = 10

        header = (
            f"{'':<{name_w}}"
            f"{'coef':>{num_w}}"
            f"{'MC sd':>{num_w}}"
            f"{'std err':>{num_w}}"
            f"{stat_label:>{num_w}}"
            f"{p_label:>{num_w}}"
            f"{f'[{q_low:.3f}':>{num_w}}"
            f"{f'{q_high:.3f}]':>{num_w}}"
            f"{f'Rej@{alpha:.2f}':>{num_w}}"
        )
        out.append(header)
        out.append(dash)

        for j, name in enumerate(param_names):
            nm = (name[:name_w]).ljust(name_w)
            row = (
                f"{nm}"
                f"{_fmt(float(b_mean[j]), digits):>{num_w}}"
                f"{_fmt(float(b_mc_sd[j]), digits):>{num_w}}"
                f"{_fmt(float(se_mean[j]), digits):>{num_w}}"
                f"{_fmt(float(stat_mean[j]), digits):>{num_w}}"
                f"{_fmt(float(p_mean[j]), digits):>{num_w}}"
                f"{_fmt(float(b_qlo[j]), digits):>{num_w}}"
                f"{_fmt(float(b_qhi[j]), digits):>{num_w}}"
                f"{_fmt(float(rej[j]), digits):>{num_w}}"
            )
            out.append(row)

        out.append(line)
        out.append(
            "Notes: 'MC sd' = std of parameter estimates across replications; "
            "'std err' = mean within-replication standard error (from vcov)."
        )
        out.append(f"      'Rej@alpha' = rejection frequency for H0: beta=0 using two-sided {test_label}.")

        cov = self.coverage(alpha=alpha, use_t=self.use_t)
        bias = self.mc_bias()
        rmse = self.mc_rmse()

        if cov is not None and bias is not None and rmse is not None:
            out.append("")
            out.append("Monte Carlo performance relative to beta_true".center(width))
            out.append(dash)

            name2_w = 14
            num2_w = 12
            level = 1.0 - alpha

            hdr2 = (
                f"{'':<{name2_w}}"
                f"{'Bias':>{num2_w}}"
                f"{'RMSE':>{num2_w}}"
                f"{f'Cover@{level:.2f}':>{num2_w}}"
            )
            out.append(hdr2)
            out.append(dash)

            for j, name in enumerate(param_names):
                nm = (name[:name2_w]).ljust(name2_w)
                out.append(
                    f"{nm}"
                    f"{_fmt(float(bias[j]), digits):>{num2_w}}"
                    f"{_fmt(float(rmse[j]), digits):>{num2_w}}"
                    f"{_fmt(float(cov[j]), digits):>{num2_w}}"
                )

            out.append(line)
            ci_label = "t-based" if self.use_t else "z-based"
            out.append(f"Coverage uses {ci_label} confidence intervals built from '{self.cov_type}' vcov.")

        if beta_null is not None:
            b0 = _coerce_beta_true(beta_null, ref=self.params, k=self.k)
            rej0 = self.rejection_rate_beta0(b0, alpha=alpha, use_t=self.use_t)

            label = "Power"
            if self.beta_true is not None:
                bt = _coerce_beta_true(self.beta_true, ref=self.params, k=self.k)
                if float(torch.max(torch.abs(bt - b0)).item()) < 1e-12:
                    label = "Size"

            out.append("")
            out.append(f"{label} for H0: beta = beta_null (componentwise)".center(width))
            out.append(dash)

            name3_w = 14
            num3_w = 12
            out.append(f"{'':<{name3_w}}{('Rej@' + str(alpha)):>{num3_w}}")
            out.append(dash)

            for j, name in enumerate(param_names):
                nm = (name[:name3_w]).ljust(name3_w)
                out.append(f"{nm}{_fmt(float(rej0[j]), digits):>{num3_w}}")

            out.append(line)
            out.append("Interpretation: if the DGP uses beta_true == beta_null, this is size; otherwise it is power.")

        return "\n".join(out)