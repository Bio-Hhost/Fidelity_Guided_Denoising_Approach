#!/usr/bin/env python3
"""
export_training_curves.py

N2V and DeepCAD-RT don't save a history file in their notebooks, so their loss
values live only in the printed training logs captured as notebook cell outputs.
This script reads those cell outputs, parses the per-epoch losses, and writes a
CSV per model so every supplement curve is backed by a data file.

Usage:
    python export_training_curves.py camp_methods_n2v.ipynb      n2v_training_curve.csv     --model n2v
    python export_training_curves.py camp_methods_DeepCAD.ipynb  deepcad_training_curve.csv --model deepcad
"""
import argparse
import csv
import json
import re


def notebook_output_text(path):
    nb = json.load(open(path, encoding="utf-8"))
    chunks = []
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            if "text" in out:  # stream output
                chunks.append("".join(out["text"]) if isinstance(out["text"], list) else out["text"])
            data = out.get("data", {})
            if "text/plain" in data:
                tp = data["text/plain"]
                chunks.append("".join(tp) if isinstance(tp, list) else tp)
    return "\n".join(chunks)


def parse_n2v(text):
    loss = re.findall(r"- loss: ([0-9.]+)", text) or re.findall(r"loss: ([0-9.]+)", text)
    val = re.findall(r"val_loss: ([0-9.]+)", text)
    n = min(len(loss), len(val))
    rows = [(i + 1, float(loss[i]), float(val[i])) for i in range(n)]
    return ["epoch", "train_loss", "val_loss"], rows


def parse_deepcad(text):
    tot = re.findall(r"Total loss: ([0-9.]+)", text)
    l1 = re.findall(r"L1 Loss: ([0-9.]+)", text)
    n = min(len(tot), len(l1))
    rows = [(i + 1, float(tot[i]), float(l1[i])) for i in range(n)]
    return ["epoch", "total_loss", "l1_loss"], rows


def training_time_seconds(text, model):
    if model == "n2v":
        secs = [float(s) for s in re.findall(r"(\d+(?:\.\d+)?)s\s+\d+m?s/step[^\r\n]*val_loss", text)]
        return (sum(secs), len(secs)) if secs else (None, 0)
    else:  # deepcad
        tc = [int(s) for s in re.findall(r"Time cost:\s*(\d+)\s*s", text)]
        return (max(tc), len(tc)) if tc else (None, 0)


def main():
    ap = argparse.ArgumentParser(description="Export training-log losses (and time) from a notebook to CSV.")
    ap.add_argument("notebook", help="Path to the .ipynb file.")
    ap.add_argument("out_csv", help="Path to write the CSV.")
    ap.add_argument("--model", choices=["n2v", "deepcad"], required=True)
    args = ap.parse_args()

    text = notebook_output_text(args.notebook)
    header, rows = (parse_n2v if args.model == "n2v" else parse_deepcad)(text)

    if not rows:
        raise SystemExit("No epoch/loss lines found — check the notebook path/format.")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Wrote {len(rows)} epochs to {args.out_csv}  (columns: {', '.join(header)})")

    total_s, n = training_time_seconds(text, args.model)
    if total_s is not None:
        print(f"Estimated training time: {total_s/60:.1f} min ({total_s:.0f} s) from {n} timing readings.")
    else:
        print("Estimated training time: not available (no timing found in the log).")


if __name__ == "__main__":
    main()
