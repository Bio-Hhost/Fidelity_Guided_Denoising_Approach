import numpy as np
from scipy.special import erf
from mle_fit import GAIN, READ_VAR_BASE   # same alpha / read-noise as the fitter


def _psf_pixels(x0, y0, s, box):
    r = box // 2
    xs = np.arange(int(round(x0)) - r, int(round(x0)) - r + box)
    ys = np.arange(int(round(y0)) - r, int(round(y0)) - r + box)
    XX, YY = np.meshgrid(xs, ys)
    sq2 = s * np.sqrt(2.0)
    Ex = 0.5 * (erf((XX + 0.5 - x0) / sq2) - erf((XX - 0.5 - x0) / sq2))
    Ey = 0.5 * (erf((YY + 0.5 - y0) / sq2) - erf((YY - 0.5 - y0) / sq2))
    return Ex * Ey, XX, YY


def crb_spot(N_ph, b_ph, s, scale=1.0, box=11, x0=0.3, y0=0.7, gain=None, read_var_base=None):
    g = GAIN if gain is None else float(gain)
    rvb = READ_VAR_BASE if read_var_base is None else float(read_var_base)
    read_var_ph = rvb * float(scale) / (g ** 2)
    P, XX, YY = _psf_pixels(x0, y0, s, box)
    mu = b_ph + N_ph * P                      # expected photons/pixel (mean)
    # sim-exact variance: only the SIGNAL is Poisson (background is a noiseless constant), + read
    var = N_ph * P + read_var_ph               # (photons^2) NO background shot term
    # derivatives of mu wrt params
    h = 1e-4
    Pp, _, _ = _psf_pixels(x0 + h, y0, s, box); Pm, _, _ = _psf_pixels(x0 - h, y0, s, box)
    dmu_dx = N_ph * (Pp - Pm) / (2 * h)
    Pp, _, _ = _psf_pixels(x0, y0 + h, s, box); Pm, _, _ = _psf_pixels(x0, y0 - h, s, box)
    dmu_dy = N_ph * (Pp - Pm) / (2 * h)
    dmu_dN = P
    dmu_db = np.ones_like(P)
    J = np.stack([dmu_dx.ravel(), dmu_dy.ravel(), dmu_dN.ravel(), dmu_db.ravel()], axis=1)  # (npix,4)
    w = 1.0 / var.ravel()
    I = J.T @ (J * w[:, None])               
    try:
        C = np.linalg.inv(I)
    except np.linalg.LinAlgError:
        return dict(crb_x=np.nan, crb_y=np.nan, crb_N=np.nan, crb_amp=np.nan)
    sx = np.sqrt(max(C[0, 0], 0.0)); sy = np.sqrt(max(C[1, 1], 0.0)); sN = np.sqrt(max(C[2, 2], 0.0))
    amp_scale = g / (2 * np.pi * s * s)        
    return dict(crb_x=float(sx), crb_y=float(sy), crb_N=float(sN), crb_amp=float(sN * amp_scale))


def crb_from_gt_row(amp_adu, bg_adu, s, scale=1.0, gain=None, read_var_base=None):
    """ GT amplitude(ADU peak), background(ADU/pixel), sigma(px) -> CRB dict."""
    g = GAIN if gain is None else float(gain)
    N_ph = amp_adu * 2.0 * np.pi * s * s / g       
    b_ph = bg_adu / g
    return crb_spot(N_ph, b_ph, s, scale=scale, gain=gain, read_var_base=read_var_base)
