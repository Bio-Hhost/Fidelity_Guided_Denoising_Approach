"""Method labels, display order and colours. Raises on a filename it cannot identify."""
from __future__ import annotations

import os
import re

# methods

NOISY = "Noisy"
N2V = "N2V"
PN2V = "PN2V"
PPN2V = "PPN2V"
DEEPCAD = "DeepCAD-RT"
L0001_T1 = "λ = 0.001 (T=1)"
L0001_T3 = "λ = 0.001 (T=3)"
L01_T1 = "λ = 0.1 (T=1)"
L01_T3 = "λ = 0.1 (T=3)"
RL = "λ = RL"
FROZEN_T1 = "frozen-λ T=1"
FROZEN_T5 = "frozen-λ T=5"

#: Display order for legends, box groups and bar charts.
METHOD_ORDER = [
    NOISY, N2V, PN2V, PPN2V, DEEPCAD,
    L0001_T1, L0001_T3, L01_T1, L01_T3, RL,
    FROZEN_T1, FROZEN_T5,
]

FIGURE4_METHODS = [NOISY, N2V, PN2V, PPN2V, DEEPCAD, L0001_T1, L01_T1, RL]


PALETTE = {
    NOISY: "#595650",          # neutral: the control, not a series identity (chroma-floor exempt)
    L0001_T1: "#13518e",       # blue family, dark   | dL 0.229, dE_normal 23.9 - all gates pass
    L0001_T3: "#2893ff",       # blue family, light
    L01_T1: "#006643",         # green family, dark  | dL 0.198, dE_normal 19.8 - all gates pass
    L01_T3: "#41a382",         # green family, light
    RL: "#d18010",             # amber family        | see PALETTE_CONFLICT below
    FROZEN_T1: "#7a4700",
    FROZEN_T5: "#995a03",
    PPN2V: "#77039e",          # purple family, dark | dL 0.074, dE_normal 16.4 - all gates pass
    PN2V: "#925ea3",           # purple family, mid
    N2V: "#ca2dff",            # purple family, light
    DEEPCAD: "#938020",        # gold, own hue
}


class UnknownMethodError(ValueError):
    """Raised when a key cannot be resolved to a paper method."""


class ProvenanceError(ValueError):
    """Raised when a file outside the canonical registry is about to enter a figure."""


_TOKEN = r"(?:^|[^a-z0-9])"
_END = r"(?:[^a-z0-9]|$)"

_METHOD_PATTERNS = [
    (rf"{_TOKEN}ppn2v{_END}", PPN2V),
    (rf"{_TOKEN}pn2v{_END}", PN2V),
    (rf"{_TOKEN}n2v{_END}", N2V),
    (rf"{_TOKEN}deepcad(?:[-_]?rt)?{_END}", DEEPCAD),
    (rf"{_TOKEN}(?:noisy|raw){_END}", NOISY),
]

_RUN_TOKENS = {
    "rlg17frozfixedt1": FROZEN_T1,
    "rlg17frozfixed": FROZEN_T5,
    "rlg17main": RL,
    "training_run_20260724-101233": FROZEN_T1,
    "training_run_20260724-080854": FROZEN_T5,
    "training_run_20260724-023428": RL,
}

_LAMBDA_LABEL = {("0.001", "1"): L0001_T1, ("0.001", "3"): L0001_T3,
                 ("0.1", "1"): L01_T1, ("0.1", "3"): L01_T3}


def _norm(key: str) -> str:
    k = os.path.basename(str(key)).strip().lower()
    for suf in ("_detailed_results.csv", ".csv", ".tif", ".tiff"):
        if k.endswith(suf):
            k = k[: -len(suf)]
    return k


def resolve_label(key: str) -> str:
    if key is None:
        raise UnknownMethodError("resolve_label(None)")
    k = _norm(key)
    if not k:
        raise UnknownMethodError(f"empty key: {key!r}")

    # 1. run tokens, longest first (must precede the generic patterns: an RL run folder
    #    name can otherwise be caught by a looser rule)
    for tok in sorted(_RUN_TOKENS, key=len, reverse=True):
        if tok in k:
            return _RUN_TOKENS[tok]

    # 2. static fidelity-weight models: lambda token + seq token, both required
    m_lam = re.search(r"(?:lambda_)?geo[=_]?(0\.001|0\.1)(?:[^0-9]|$)", k)
    if m_lam:
        m_seq = re.search(r"(?:seq|sequence_length=)(\d+)", k)
        if not m_seq:
            raise UnknownMethodError(
                f"fidelity weight {m_lam.group(1)} found but no seq(N) token in {key!r}; "
                "the temporal window must come from the filename")
        pair = (m_lam.group(1), m_seq.group(1))
        if pair not in _LAMBDA_LABEL:
            raise UnknownMethodError(f"unsupported (lambda, T) combination {pair} in {key!r}")
        return _LAMBDA_LABEL[pair]

    # 3. baselines and noisy, anchored, longest pattern first
    for pat, lab in sorted(_METHOD_PATTERNS, key=lambda x: -len(x[0])):
        if re.search(pat, k):
            return lab

    raise UnknownMethodError(
        f"unrecognised method key {key!r} (normalised {k!r}). Add an explicit pattern to "
        "figure_style._METHOD_PATTERNS or _RUN_TOKENS -- do not add a fallback.")


def style_for_data_type(data_type_key: str):
    key = str(data_type_key)
    is_mle = key.lower().endswith("_mle")
    if is_mle:
        key = key[: -len("_MLE")]
    label = resolve_label(key)
    colour = color_for(label)
    if is_mle:
        return f"{label} (MLE)", _darken(colour)
    return label, colour


def color_for(label: str) -> str:
    if label not in PALETTE:
        raise UnknownMethodError(f"no colour registered for {label!r}; add one to PALETTE")
    return PALETTE[label]
