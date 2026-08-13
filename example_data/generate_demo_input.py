import numpy as np
import tifffile

rng = np.random.default_rng(20260703)

N_FRAMES = 30
H = W = 128 # square -> compatible with a square static model after padding
MARGIN = 26 # keep spots away from the 20 px corner background regions

OFFSET        = 100.0 # camera offset (ADU)
GAIN          = 2.2 # ADU per photoelectron (photon-transfer slope)
READ_SIGMA    = 3.0 # read noise std (ADU)  -> variance 9.0
BG_ELECTRONS  = 8.0 # optical background (photoelectrons/pixel)

N_SPOTS       = 22
PSF_SIGMA     = 1.3 # diffraction-limited spot width (px)
AMP_E_RANGE   = (40.0, 240.0) # per-spot peak signal (photoelectrons)
P_ON          = 0.75 # per-frame probability an emitter is "on" (blinking)

def gaussian_spot(h, w, x0, y0, amp, sigma):
    yy, xx = np.mgrid[0:h, 0:w]
    return amp * np.exp(-(((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma ** 2)))

spot_x = rng.uniform(MARGIN, W - MARGIN, size=N_SPOTS)
spot_y = rng.uniform(MARGIN, H - MARGIN, size=N_SPOTS)
spot_a = rng.uniform(*AMP_E_RANGE, size=N_SPOTS)

frames = np.empty((N_FRAMES, H, W), dtype=np.uint16)
for t in range(N_FRAMES):
    lam_e = np.full((H, W), BG_ELECTRONS, dtype=np.float64)
    on = rng.random(N_SPOTS) < P_ON
    for k in np.where(on)[0]:
        lam_e += gaussian_spot(H, W, spot_x[k], spot_y[k], spot_a[k], PSF_SIGMA)

    n_e = rng.poisson(lam_e).astype(np.float64)                 # shot noise
    adu = OFFSET + GAIN * n_e + rng.normal(0.0, READ_SIGMA, (H, W))  # gain + read noise
    frames[t] = np.clip(np.round(adu), 0, 65535).astype(np.uint16)

tifffile.imwrite("demo_raw_video.tif", frames, imagej=True, metadata={"axes": "TYX"})

corners = np.concatenate([
    frames[:, 0:20, 0:20].ravel(),   frames[:, 0:20, W-20:W].ravel(),
    frames[:, H-20:H, 0:20].ravel(), frames[:, H-20:H, W-20:W].ravel(),
]).astype(np.float64)
bg_level = float(np.median(corners))

noise_params = {
    "background_level":  bg_level,
    "gaussian_variance": float(READ_SIGMA ** 2),
    "gain_estimate":     float(GAIN),
}
np.save("demo_noise_params.npy", noise_params)

print("Wrote demo_raw_video.tif  shape=%s dtype=%s" % (frames.shape, frames.dtype))
print("Wrote demo_noise_params.npy ->", noise_params)
print("Frame intensity range: [%d, %d]" % (frames.min(), frames.max()))
print("Recovered background (corner median): %.2f  (truth offset+gain*bg = %.2f)"
      % (bg_level, OFFSET + GAIN * BG_ELECTRONS))
