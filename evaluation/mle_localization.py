"""
mle_localization.py  -- noise-aware MLE localization.

Designed to drop into evaluate_full.py: the only change needed there is to make
the fitter in evaluate_video() swappable. Then the four arms are

    raw-LSQ / denoised-LSQ  -> evaluate_full.fit_rotated_gaussian_2d  (unchanged)
    raw-MLE / denoised-MLE  -> make_mle_fitter(K, read_var, bg)       (this file)

All arms then flow through the SAME detection -> GT-match -> fit -> metrics
pipeline, so Loc_RMSE / Loc_MedianAE / Phot_* / PSNR / SSIM / F1 are directly
comparable and land in the same master CSV and plots.

Simulation noise model (add_noise_to_gt.py):
    y_i = bg + Poisson(K * s_i) / K + N(0, sigma_read^2),   s_i = signal above bg
=>  Var_i = K * (mu_i - bg) + sigma_read^2      (the MLE's per-pixel weights)

For the denoised-MLE arm, passing the RAW (K, read_var, bg) is intentionally
misspecified -- it shows that denoised images need their own
noise model. The parameters are arguments so an empirical denoised-noise model
can be supplied instead.
"""
import numpy as np
from scipy.optimize import curve_fit

# fit-param dict keys must match what evaluate_full.py consumes
#   (result_entry.update(params); metrics use fit_x, fit_y, fit_amplitude)


def rotated_2d_gaussian(coords, amplitude, x0, y0, sigma_x, sigma_y, theta_deg, offset):
    (x, y) = coords
    theta = np.deg2rad(theta_deg)
    xc, yc = x - x0, y - y0
    xp = xc * np.cos(theta) + yc * np.sin(theta)
    yp = -xc * np.sin(theta) + yc * np.cos(theta)
    sx2, sy2 = sigma_x ** 2, sigma_y ** 2
    exponent = (xp ** 2) / (2 * sx2 + 1e-7) + (yp ** 2) / (2 * sy2 + 1e-7)
    return offset + amplitude * np.exp(-exponent)


def _mesh(region, gx1, gy1):
    h, w = region.shape
    Y, X = np.mgrid[gy1:gy1 + h, gx1:gx1 + w]
    return X, Y


def _core_fit(region, gx1, gy1, sigma=None, p0=None):
    h, w = region.shape
    if h * w < 8:
        return False, None
    X, Y = _mesh(region, gx1, gy1)
    z = region.ravel().astype(float)
    if p0 is None:
        vmin, vmax = float(region.min()), float(region.max())
        p0 = (max(vmax - vmin, 1e-6), gx1 + w / 2.0, gy1 + h / 2.0, 1.0, 1.0, 0.0, vmin)
    bounds = ([0, -np.inf, -np.inf, 0.2, 0.2, -180, -np.inf],
              [np.inf, np.inf, np.inf, w, h, 180, np.inf])

    def f(coords, *a):
        return rotated_2d_gaussian(coords, *a).ravel()

    kw = dict(p0=p0, bounds=bounds, maxfev=5000, method='trf', ftol=1e-4, xtol=1e-4)
    if sigma is not None:
        kw['sigma'] = sigma.ravel()
        kw['absolute_sigma'] = True
    try:
        popt, _ = curve_fit(f, (X, Y), z, **kw)
    except Exception:
        return False, None
    return True, popt


def _pack(popt):
    return {'fit_amplitude': popt[0], 'fit_x': popt[1], 'fit_y': popt[2],
            'fit_sx': popt[3], 'fit_sy': popt[4], 'fit_theta': popt[5], 'fit_offset': popt[6]}


def _valid(popt):
    return popt[0] > 0 and popt[3] >= 0.2 and popt[4] >= 0.2


def fit_gaussian_lsq(region, gx1, gy1):
    ok, popt = _core_fit(region, gx1, gy1, sigma=None)
    if not ok or not _valid(popt):
        return False, None
    return True, _pack(popt)


def fit_gaussian_mle(region, gx1, gy1, gain_adu_per_photon, read_var, background,
                     n_irls=4, max_drift=None):
    ok, popt = _core_fit(region, gx1, gy1, sigma=None)   # OLS start
    if not ok:
        return False, None
    h, w = region.shape
    if max_drift is None:
        max_drift = 0.25 * min(h, w)
    x0_ols, y0_ols = popt[1], popt[2]
    X, Y = _mesh(region, gx1, gy1)
    for _ in range(n_irls):
        mu = rotated_2d_gaussian((X, Y), *popt)
        signal = np.maximum(mu - background, 0.0)
        var = gain_adu_per_photon * signal + read_var
        sig = np.sqrt(np.maximum(var, 1e-6))
        ok_new, popt_new = _core_fit(region, gx1, gy1, sigma=sig, p0=popt)
        if not ok_new:
            break
        drift = np.hypot(popt_new[1] - x0_ols, popt_new[2] - y0_ols)
        if drift > max_drift:          # reweighting diverged -> keep last stable fit
            break
        converged = np.allclose(popt_new[1:3], popt[1:3], atol=1e-4)
        popt = popt_new
        if converged:
            break
    if not _valid(popt):
        return False, None
    return True, _pack(popt)


def make_mle_fitter(gain_adu_per_photon, read_var, background, **kw):
    def fitter(region, gx1, gy1):
        return fit_gaussian_mle(region, gx1, gy1, gain_adu_per_photon,
                                read_var, background, **kw)
    return fitter


def crlb_xy(theta, region_shape, gx1, gy1, gain_adu_per_photon, read_var, background):
    X, Y = _mesh(np.empty(region_shape), gx1, gy1)
    p = np.array(theta, float)

    def mu_of(pv):
        return rotated_2d_gaussian((X, Y), *pv).ravel()

    signal = np.maximum(mu_of(p) - background, 0.0)
    inv_var = 1.0 / np.maximum(gain_adu_per_photon * signal + read_var, 1e-9)
    J = np.zeros((mu_of(p).size, len(p)))
    for j in range(len(p)):
        step = 1e-4 * max(abs(p[j]), 1.0)
        pp, pm = p.copy(), p.copy(); pp[j] += step; pm[j] -= step
        J[:, j] = (mu_of(pp) - mu_of(pm)) / (2 * step)
    try:
        cov = np.linalg.inv(J.T @ (inv_var[:, None] * J))
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    return np.sqrt(max(cov[1, 1], 0)), np.sqrt(max(cov[2, 2], 0))


def _draw(true_params, shape, gx1, gy1, K, sigma_read, bg, rng):
    X, Y = _mesh(np.empty(shape), gx1, gy1)
    clean = rotated_2d_gaussian((X, Y), *true_params)
    signal = np.maximum(clean - bg, 0.0)
    shot = rng.poisson(signal / K) * K
    return bg + shot + rng.normal(0.0, sigma_read, size=shape)


def _robust_std(errs, tol):
    e = np.array(errs)
    ok = np.abs(e) < tol
    if ok.sum() < 5:
        return np.nan, ok.mean() if e.size else 0.0
    kept = e[ok]
    return 1.4826 * np.median(np.abs(kept - np.median(kept))), ok.mean()


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    K, SIGMA_READ, BG = 18.0, 19.7, 198.0
    READ_VAR = SIGMA_READ ** 2
    shape, (gx1, gy1) = (15, 15), (0, 0)
    x_true, y_true, s_true = 7.35, 6.80, 1.4
    TOL, N_MC = 3.0, 400

    print(f"{'amp':>6} {'pkSNR':>6} | {'LSQ rMAD':>9} {'succ':>5} | {'MLE rMAD':>9} {'succ':>5} | "
          f"{'CRLB':>7} | {'eff_LSQ':>7} {'eff_MLE':>7}")
    for amp in (60.0, 150.0, 400.0, 1000.0):
        tp = (amp, x_true, y_true, s_true, s_true, 0.0, BG)
        pk = amp / np.sqrt(K * amp + READ_VAR)
        le, me = [], []
        for _ in range(N_MC):
            roi = _draw(tp, shape, gx1, gy1, K, SIGMA_READ, BG, rng)
            ok, p = fit_gaussian_lsq(roi, gx1, gy1)
            le.append(p['fit_x'] - x_true if ok else 1e9)
            ok, p = fit_gaussian_mle(roi, gx1, gy1, K, READ_VAR, BG)
            me.append(p['fit_x'] - x_true if ok else 1e9)
        lstd, lsucc = _robust_std(le, TOL)
        mstd, msucc = _robust_std(me, TOL)
        cx, _ = crlb_xy(tp, shape, gx1, gy1, K, READ_VAR, BG)
        el = cx / lstd if lstd == lstd else np.nan
        em = cx / mstd if mstd == mstd else np.nan
        print(f"{amp:6.0f} {pk:6.2f} | {lstd:9.4f} {lsucc:5.2f} | {mstd:9.4f} {msucc:5.2f} | "
              f"{cx:7.4f} | {el:7.2f} {em:7.2f}")
    print("\nrMAD = robust (MAD-based) precision over successful fits; eff = CRLB/rMAD (->1 is optimal).")
