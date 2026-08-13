"""Poisson-Gaussian maximum-likelihood 2D-Gaussian fitter.

Shares the EXACT forward model, initial guess, and bounds with the LSE fitter
(`evaluation/evaluate_full.py:fit_rotated_gaussian_2d`). Only the objective differs: instead of
least-squares residuals, we minimise the negative Poisson-Gaussian log-likelihood of the observed
patch under the model, using the same gain and read-noise the paper's training loss uses.
"""
import os, sys, numpy as np
from scipy.optimize import minimize

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL = os.path.join(_HERE, "evaluation")
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)
from evaluate_full import rotated_2d_gaussian

# noise parameters: same source the trainers use (filtered-PTC gain 17.689)
_NP = np.load(os.path.join(_HERE, "trained_models", "static_models_new",
              "static_T1_geo0.1_20260723-202607", "noise_parameters.npy"), allow_pickle=True).item()
GAIN = float(_NP["gain_estimate"])          # alpha, ADU/photon; Cy3 (17.689). v2 sim: 18.052
READ_VAR_BASE = float(_NP["gaussian_variance"])  # read-noise var, ADU^2; Cy3 (316.527). v2 sim: 387.205


def _init_and_bounds(region, gx1, gy1):
    h, w = region.shape
    min_r, max_r = float(np.min(region)), float(np.max(region))
    amp_g = (max_r - min_r) if max_r > min_r else 1.0
    off_g = min_r
    x0_g = gx1 + w / 2.0
    y0_g = gy1 + h / 2.0
    sig_g = max(1.0, min(w, h) / 4.0)
    init = np.array([amp_g, x0_g, y0_g, sig_g, sig_g, 0.0, off_g], float)
    lo = [0.0, gx1 - 2, gy1 - 2, 0.1, 0.1, -180.0, -np.inf]
    hi = [np.inf, gx1 + w + 2, gy1 + h + 2, w * 2.0, h * 2.0, 180.0, np.inf]
    return init, lo, hi


def _pg_nll(params, X, Y, y_obs, read_var, gain):
    m = rotated_2d_gaussian((X, Y), *params)
    signal = np.clip(m - params[6], 0.0, None) 
    V = gain * signal + read_var + 1e-9 
    r = y_obs - m
    return 0.5 * np.sum(r * r / V + np.log(V))


def fit_mle_gaussian_2d(region, gx1, gy1, scale=1.0, gain=None, read_var_base=None):
    g = GAIN if gain is None else float(gain)
    rvb = READ_VAR_BASE if read_var_base is None else float(read_var_base)
    h, w = region.shape
    if h == 0 or w == 0:
        return False, None
    Y, X = np.mgrid[gy1:gy1 + h, gx1:gx1 + w]
    y_obs = region.astype(float)
    read_var = rvb * float(scale)
    init, lo, hi = _init_and_bounds(region, gx1, gy1)
    b_lbfgs = [(a if np.isfinite(a) else None, c if np.isfinite(c) else None) for a, c in zip(lo, hi)]
    p = None
    try:
        res = minimize(_pg_nll, init, args=(X, Y, y_obs, read_var, g), method="L-BFGS-B", bounds=b_lbfgs)
        if res.success and np.all(np.isfinite(res.x)):
            p = res.x
    except Exception:
        p = None
    if p is None:
        try:
            res = minimize(_pg_nll, init, args=(X, Y, y_obs, read_var, g), method="Nelder-Mead",
                           options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-4})
            lo_a = np.array([a if np.isfinite(a) else -1e18 for a in lo])
            hi_a = np.array([c if np.isfinite(c) else 1e18 for c in hi])
            p = np.clip(res.x, lo_a, hi_a)
            if not np.all(np.isfinite(p)):
                return False, None
        except Exception:
            return False, None
    return True, {"fit_x": float(p[1]), "fit_y": float(p[2]), "fit_amplitude": float(p[0]),
                  "fit_sx": float(p[3]), "fit_sy": float(p[4]), "fit_theta": float(p[5]),
                  "fit_offset": float(p[6])}
