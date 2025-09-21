#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Retune a *single* label on calibrated VAL probabilities subject to:
- target PPV (point) AND Wilson LB >= target_ppv
- min_recall, min_cov, max_calib_gap
- AND binomial p-value <= alpha (right-tailed test vs p0=target_ppv)

Select τ maximizing recall (then coverage). Report TEST at τ.
Optionally commit τ into tuned_thresholds.npy.
"""
import os, argparse, numpy as np
from common_safety import (
    load_split_arrays, load_label_order, load_thresholds, safe_update_threshold,
    wilson_lower_bound, binom_sf_geq
)

def grid_from_string(spec: str) -> np.ndarray:
    a,b,c = [float(x) for x in spec.split(":")]
    n = int(np.floor((b - a) / c)) + 1
    return np.round(a + np.arange(n)*c, 6)

def eval_at(y, m, p, thr, alpha, target):
    mask = m.astype(bool); yv = y[mask].astype(int); pv = p[mask]
    pred = (pv >= thr).astype(np.int8)
    tp = int(((yv == 1) & (pred == 1)).sum())
    fp = int(((yv == 0) & (pred == 1)).sum())
    fn = int(((yv == 1) & (pred == 0)).sum())
    npos = int(pred.sum())
    prec = (tp / npos) if npos > 0 else 0.0
    rec  = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    cov  = npos / max(1, yv.size)
    wlb  = wilson_lower_bound(tp, npos, alpha=alpha)
    mean_conf = float(pv[pred==1].mean()) if npos>0 else 0.0
    gap = abs(prec - mean_conf)
    pval = binom_sf_geq(tp, npos, target) if npos>0 else 1.0
    return dict(tp=tp, fp=fp, fn=fn, npos=npos, precision=prec, recall=rec,
                coverage=cov, wilson_lb=wlb, calib_gap=gap, pval=pval)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--target_ppv", type=float, default=0.90)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--min_recall", type=float, default=0.20)
    ap.add_argument("--min_cov", type=float, default=0.05)
    ap.add_argument("--max_calib_gap", type=float, default=0.08)
    ap.add_argument("--grid", type=str, default="0.80:0.99:0.001")
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    labels = load_label_order(args.run_dir)
    if args.label not in labels:
        raise ValueError("label not found")
    j = labels.index(args.label)

    # prefer calibrated if present
    yv, mv, pv = load_split_arrays(args.run_dir, "val")
    yt, mt, pt = load_split_arrays(args.run_dir, "test")

    grid = grid_from_string(args.grid)
    best_t, best_ev = None, None

    print("=== Calibrated Safety + p-value Retune ===")
    for t in grid:
        ev = eval_at(yv[:, j], mv[:, j], pv[:, j], float(t), args.alpha, args.target_ppv)
        ok = (ev["precision"] >= args.target_ppv and ev["wilson_lb"] >= args.target_ppv
              and ev["recall"] >= args.min_recall and ev["coverage"] >= args.min_cov
              and ev["calib_gap"] <= args.max_calib_gap and ev["pval"] <= args.alpha)
        if ok and (best_ev is None or ev["recall"] > best_ev["recall"] or
                   (ev["recall"] == best_ev["recall"] and ev["coverage"] > best_ev["coverage"])):
            best_t, best_ev = float(t), ev

    if best_t is None:
        # print feasibility guidance at nearby PredPos
        print("No threshold found that meets safety + p<=alpha. Consider adjusting alpha or target_ppv.")
        # derive quick table around current positives
        for npos in range(70, 101):
            # minimal TP so that p-value <= alpha is a bit involved; just print PPV floor needed
            need_ppv = max(args.target_ppv, (np.ceil((args.target_ppv*npos)) / max(1,npos)))
            print(f"PredPos={npos}: need TP ≥ {int(np.ceil(need_ppv*npos))} (PPV ≥ {need_ppv:.3f})")
        return

    evT = eval_at(yt[:, j], mt[:, j], pt[:, j], best_t, args.alpha, args.target_ppv)

    print(f"Chosen thr (VAL): {best_t:.3f}")
    print(f"VAL:  PPV={best_ev['precision']:.3f}  WilsonLB={best_ev['wilson_lb']:.3f}  "
          f"Recall={best_ev['recall']:.3f}  Coverage={best_ev['coverage']:.3f}  "
          f"CalibGap={best_ev['calib_gap']:.3f}  TP={best_ev['tp']}  PredPos={best_ev['npos']}  p={best_ev['pval']:.4g}")
    print(f"TEST: PPV={evT['precision']:.3f}  WilsonLB={evT['wilson_lb']:.3f}  "
          f"Recall={evT['recall']:.3f}  Coverage={evT['coverage']:.3f}  "
          f"CalibGap={evT['calib_gap']:.3f}  TP={evT['tp']}  PredPos={evT['npos']}")

    if args.commit:
        out = safe_update_threshold(args.run_dir, {args.label: best_t}, labels, commit=True)
        print(f"[INFO] Updated tuned_thresholds[{args.label}] -> {best_t:.3f} at {out}")

if __name__ == "__main__":
    main()
