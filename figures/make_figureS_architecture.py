import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINER = os.path.join(ROOT, "train_lambda_RL.py")
STATIC_TRAINER = os.path.join(ROOT, "train_static_lambda.py")
OUTPDF = os.path.join(ROOT, "figures", "figureS_architecture.pdf")
OUTPNG = os.path.join(ROOT, "figures", "figureS_architecture.png")

FRAME_PX = 256
SEQ = 5

C_ENC = "#ffffff"       # encoder / decoder blocks
C_BOT = "#c9c6be"       # bottleneck
C_OUT = "#8a877f"       # the single output frame
INK = "#1a1a19"


def load_builder(path):
    import tensorflow as tf
    spec = importlib.util.spec_from_file_location("trainer_" + os.path.basename(path), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod.build_3d_unet


def read_model():
    from tensorflow.keras import layers
    build = load_builder(TRAINER)
    m = build((FRAME_PX, FRAME_PX, SEQ, 1), SEQ)

    conv = [l for l in m.layers
            if isinstance(l, layers.Conv3D) and not isinstance(l, layers.Conv3DTranspose)]
    convT = [l for l in m.layers if isinstance(l, layers.Conv3DTranspose)]
    pools = [l for l in m.layers if isinstance(l, layers.MaxPool3D)]
    twod = [l for l in m.layers
            if isinstance(l, (layers.Conv2D, layers.Conv2DTranspose, layers.MaxPool2D))]

    kern = {}
    for l in conv:
        kern[tuple(l.kernel_size)] = kern.get(tuple(l.kernel_size), 0) + 1
    pool_windows = {tuple(l.pool_size) for l in pools}
    if len(pool_windows) != 1:
        raise RuntimeError(f"pooling windows are not uniform: {pool_windows}")

    def shp(layer):
        s = tuple(layer.output.shape)
        return dict(h=int(s[1]), w=int(s[2]), t=int(s[3]), c=int(s[4]) if len(s) > 4 else 1)

    enc, dec = [], []
    seen_pool = 0
    for l in m.layers:
        if isinstance(l, layers.MaxPool3D):
            seen_pool += 1
    order = [l for l in m.layers if isinstance(l, (layers.Conv3D, layers.MaxPool3D))
             and not isinstance(l, layers.Conv3DTranspose)]
    convs_in_order = [l for l in order if not isinstance(l, layers.MaxPool3D)]
    for i in range(3):
        enc.append(shp(convs_in_order[2 * i + 1]))
    bott = shp(convs_in_order[7])
    for i in range(3):
        dec.append(shp(convs_in_order[9 + 2 * i]))
    proj = convs_in_order[14]
    out = shp(m.layers[-1]) if len(m.layers[-1].output.shape) > 4 else None
    out_shape = tuple(int(x) for x in m.layers[-1].output.shape[1:])

    static_params = None
    try:
        sbuild = load_builder(STATIC_TRAINER)
        static_params = int(sbuild((FRAME_PX, FRAME_PX, SEQ, 1), SEQ).count_params())
    except Exception as e:
        print(f"  static comparator unavailable: {e}")

    return dict(
        model=m, enc=enc, bott=bott, dec=dec,
        n_conv=len(conv), n_convT=len(convT), n_pool=len(pools), n_2d=len(twod),
        kernels={"x".join(map(str, k)): v for k, v in kern.items()},
        transpose_kernels={"x".join(map(str, tuple(l.kernel_size))) for l in convT},
        transpose_strides={"x".join(map(str, tuple(l.strides))) for l in convT},
        pool_window=next(iter(pool_windows)),
        proj_kernel=tuple(proj.kernel_size), proj_filters=int(proj.filters),
        params=int(m.count_params()), static_params=static_params,
        out_shape=out_shape,
        rep_tensors=[tuple(int(x) for x in l.kernel.shape) for l in conv[:3]],
    )


def main():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    M = read_model()
    pw = "x".join(map(str, M["pool_window"]))
    print(f"built from {os.path.basename(TRAINER)}: {M['n_conv']} Conv3D "
          f"({M['kernels']}), {M['n_convT']} Conv3DTranspose, {M['n_pool']} MaxPool3D {pw}, "
          f"{M['n_2d']} 2D layers, {M['params']:,} parameters")
    print(f"  encoder  {[ (e['h'], e['t'], e['c']) for e in M['enc'] ]}")
    print(f"  bottleneck {(M['bott']['h'], M['bott']['t'], M['bott']['c'])}")
    print(f"  decoder  {[ (d['h'], d['t'], d['c']) for d in M['dec'] ]}")
    print(f"  output   {M['out_shape']}")
    if M["static_params"]:
        print(f"  static family: {M['static_params']:,} parameters "
              f"({M['params'] - M['static_params']:,} fewer)")

    fig = plt.figure(figsize=(6.75, 3.30))
    ax = fig.add_axes([0.005, 0.010, 0.99, 0.980]); ax.set_axis_off()
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    LEVEL_Y = {0: 76, 1: 57, 2: 38, 3: 19}
    ENC_X = [11, 25, 39]
    BOT_X = 51
    DEC_X = [63, 76, 87]
    BLOCK_W, BLOCK_H = 6.6, 11.0
    DX, DY = 0.75, 0.95

    def stack(xc, yc, t, fill, chan, spatial):
        """A block drawn as t offset slices, so temporal depth is visible as a dimension."""
        for i in range(t - 1, -1, -1):
            ax.add_patch(Rectangle((xc - BLOCK_W / 2 + i * DX, yc - BLOCK_H / 2 + i * DY),
                                   BLOCK_W, BLOCK_H, facecolor=fill, edgecolor=INK, lw=0.7,
                                   zorder=3 + i))
        mx = xc + (t - 1) * DX / 2
        top = yc + BLOCK_H / 2 + (t - 1) * DY
        bot = yc - BLOCK_H / 2
        ax.text(mx, top + 1.3, chan, ha="center", va="bottom", fontsize=7, color=INK, zorder=20)
        ax.text(mx, bot - 1.5, spatial, ha="center", va="top", fontsize=7, color="#57544e",
                zorder=20)
        return dict(left=xc - BLOCK_W / 2, right=xc + BLOCK_W / 2 + (t - 1) * DX,
                    mid_y=yc + (t - 1) * DY / 2, top=top, bot=bot, mx=mx)

    B = {}
    for i, e in enumerate(M["enc"]):
        B[("enc", i)] = stack(ENC_X[i], LEVEL_Y[i], e["t"], C_ENC, f"{e['c']}",
                              f"{e['h']}x{e['w']}")
    b = M["bott"]
    B[("bot", 0)] = stack(BOT_X, LEVEL_Y[3], b["t"], C_BOT, f"{b['c']}", f"{b['h']}x{b['w']}")
    for i, d in enumerate(M["dec"]):
        B[("dec", i)] = stack(DEC_X[i], LEVEL_Y[2 - i], d["t"], C_ENC, f"{d['c']}",
                              f"{d['h']}x{d['w']}")

    def arrow(p0, p1, style="-", colour=INK, lw=0.9, rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=lw,
                                     color=colour, linestyle=style, zorder=2,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=1.0, shrinkB=1.0))

    chain = [("enc", 0), ("enc", 1), ("enc", 2), ("bot", 0), ("dec", 0), ("dec", 1), ("dec", 2)]
    for a_, b_ in zip(chain[:-1], chain[1:]):
        arrow((B[a_]["right"] + 0.6, B[a_]["mid_y"]), (B[b_]["left"] - 0.6, B[b_]["mid_y"]))

    for i in range(3):
        a_, b_ = B[("enc", i)], B[("dec", 2 - i)]
        arrow((a_["mx"], a_["top"] + 4.2), (b_["mx"], b_["top"] + 4.2),
              style=(0, (3, 2)), colour="#7a7770", lw=0.8, rad=-0.10)

    ax.text(1.0, LEVEL_Y[0] + BLOCK_H / 2 + 12.0,
            "input  %dx%dx%d" % (FRAME_PX, FRAME_PX, SEQ),
            ha="left", va="bottom", fontsize=7, color=INK)
    arrow((4.0, LEVEL_Y[0] + BLOCK_H / 2 + 11.2),
          (B[("enc", 0)]["left"] + 1.0, B[("enc", 0)]["top"] + 0.8), rad=0.18)

    ox = 97.0
    last = B[("dec", 2)]
    ax.add_patch(Rectangle((ox - 2.3, LEVEL_Y[0] - BLOCK_H / 2), 4.6, BLOCK_H,
                           facecolor=C_OUT, edgecolor=INK, lw=0.7, zorder=8))
    ax.text(ox, LEVEL_Y[0] - BLOCK_H / 2 - 1.5,
            "%dx%dx1" % (M["out_shape"][0], M["out_shape"][1]),
            ha="center", va="top", fontsize=7, color="#57544e")
    arrow((last["right"] + 0.6, last["mid_y"]), (ox - 2.9, LEVEL_Y[0]), lw=1.1)

    kern3 = [k for k in M["kernels"] if k != "1x1x1"][0]
    sp = lambda t: t.replace("x", " x ")
    handles = [
        Patch(facecolor=C_ENC, edgecolor=INK, lw=0.7,
              label="encoder / decoder block: 2 x Conv3D " + sp(kern3)),
        Patch(facecolor=C_BOT, edgecolor=INK, lw=0.7,
              label="bottleneck: 2 x Conv3D " + sp(kern3)),
        Patch(facecolor=C_OUT, edgecolor=INK, lw=0.7,
              label="output frame: " + sp("x".join(map(str, M["proj_kernel"])))
                    + " projection, then central-frame extraction"),
        Line2D([], [], color=INK, lw=0.9,
               label="max pool " + sp(pw) + " (down) / Conv3DTranspose stride "
                     + sp(pw) + " (up)"),
        Line2D([], [], color="#7a7770", lw=0.8, ls=(0, (3, 2)),
               label="skip connection (concatenation)"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=1,
              frameon=False, fontsize=7, handlelength=1.9, labelspacing=0.40,
              handletextpad=0.7, borderaxespad=0.0)

    fig.savefig(OUTPDF); fig.savefig(OUTPNG, dpi=400, facecolor="white"); plt.close(fig)
    print(f"wrote {os.path.relpath(OUTPDF, ROOT)} and .png")


if __name__ == "__main__":
    main()
