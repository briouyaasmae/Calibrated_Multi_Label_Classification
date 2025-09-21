#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimize one primary label's operating threshold on VAL with safety constraints,
report TEST metrics, and (optionally) commit threshold to tuned_thresholds.npy.

Constraints:
- target_ppv (point) AND Wilson LB >= target_ppv
- min_recall, min_cov, max_calib_gap

Supports multi-label FDR (BH) using per-label one-sided binomial tests at target_ppv.
"""
import os, json, argparse
import numpy as np
from common_safety import (
    ensure_dir, load_split_arrays, load_label_order, load_thresholds, safe_update_threshold,
    confusion_at, wilson_lower_bound, binom_sf_geq, fdr_keep_mask
)

def grid_from_string(spec: str) -> np.ndarray:
    # "0.30:0.99:0.005"
    a,b,c = [float(x) for x in spec.split(":")]
    n = int(np.floor((b - a) / c)) + 1
    return np.round(a + np.arange(n)*c, 6)

def evaluate_label(y, m, p, thr, alpha, target_ppv):
    mask = m.astype(bool)
    yv, pv = y[mask].astype(int), p[mask]
    tp, fp, fn, npos = confusion_at(pv, yv, thr)
    prec = (tp / npos) if npos > 0 else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    cov = npos / max(1, yv.size)
    wlb = wilson_lower_bound(tp, npos, alpha=alpha)
    mean_conf = float(pv[pv >= thr].mean()) if npos > 0 else 0.0
    calib_gap = abs(prec - mean_conf)
    return dict(tp=tp, fp=fp, fn=fn, npos=npos, precision=prec, recall=rec,
                coverage=cov, wilson_lb=wlb, mean_conf=mean_conf, calib_gap=calib_gap)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--primary_label", required=True)
    ap.add_argument("--target_ppv", type=float, default=0.85)
    ap.add_argument("--alpha", type=float, default=0.15, help="one-sided")
    ap.add_argument("--criterion", choices=["wilson"], default="wilson")
    ap.add_argument("--grid", type=str, default="0.30:0.99:0.005")
    ap.add_argument("--min_recall", type=float, default=0.20)
    ap.add_argument("--min_cov", type=float, default=0.05)
    ap.add_argument("--max_calib_gap", type=float, default=0.08)
    ap.add_argument("--safety_labels", type=str, default="", help="comma separated; if set we do FDR")
    ap.add_argument("--fdr_method", choices=["bh","none"], default="bh")
    ap.add_argument("--commit", action="store_true", help="actually write tuned_thresholds.npy")
    args = ap.parse_args()

    print("=== Safety-Aware Deployment Checklist ===")
    print(f"Run Dir            : {args.run_dir}")
    print(f"Primary Label      : {args.primary_label}")
    print(f"Target PPV         : {args.target_ppv:.2f}  (alpha_one_sided={args.alpha:.2f}, criterion={args.criterion})")

    # load
    yv, mv, pv = load_split_arrays(args.run_dir, "val")
    yt, mt, pt = load_split_arrays(args.run_dir, "test")
    labels = load_label_order(args.run_dir)
    if args.primary_label not in labels:
        raise ValueError("primary_label not found")
    j = labels.index(args.primary_label)

    base_thr = load_thresholds(args.run_dir, len(labels))[j]
    grid = grid_from_string(args.grid)

    # search best by: maximize recall subject to constraints (could also maximize coverage)
    best_thr = None; best_val = None
    for t in grid:
        ev = evaluate_label(yv[:, j], mv[:, j], pv[:, j], float(t), args.alpha, args.target_ppv)
        ok = (ev["precision"] >= args.target_ppv and ev["wilson_lb"] >= args.target_ppv
              and ev["recall"] >= args.min_recall and ev["coverage"] >= args.min_cov
              and ev["calib_gap"] <= args.max_calib_gap)
        if ok and (best_thr is None or ev["recall"] > best_val["recall"] or
                   (ev["recall"] == best_val["recall"] and ev["coverage"] > best_val["coverage"])):
            best_thr, best_val = float(t), ev

    if best_thr is None:
        # pick the t with max Wilson LB (diagnostic)
        wlbs = []
        for t in grid:
            ev = evaluate_label(yv[:, j], mv[:, j], pv[:, j], float(t), args.alpha, args.target_ppv)
            wlbs.append((ev["wilson_lb"], float(t), ev))
        wlbs.sort(reverse=True, key=lambda x: x[0])
        best_thr = wlbs[0][1]; best_val = wlbs[0][2]
        print("\n=== RESULT (VAL cannot meet safety constraints) ===")
        print(f"VAL cannot meet target constraints.\n"
              f"Target PPV (criterion={args.criterion}): {args.target_ppv:.2f} | "
              f"min_recall: {args.min_recall:.2f} | min_cov: {args.min_cov:.2f} | max_calib_gap: {args.max_calib_gap:.2f}")
        print(f"Best Wilson LB on VAL: {wlbs[0][0]:.3f} at thr={best_thr:.3f}\n")
    else:
        print("\n--- Recommended Threshold (VAL) ---")
        print(f"thr={best_thr:.3f} | PPV={best_val['precision']:.3f} | WilsonLB={best_val['wilson_lb']:.3f} | "
              f"Recall={best_val['recall']:.3f} | Coverage={best_val['coverage']:.3f} | "
              f"CalibGap={best_val['calib_gap']:.3f} | FP/1k={best_val['fp'] / max(1,(mv[:, j].sum())) * 1000:.3f}")

    # TEST at best_thr
    et = evaluate_label(yt[:, j], mt[:, j], pt[:, j], best_thr, args.alpha, args.target_ppv)
    print("\n--- TEST at Recommended Threshold ---")
    print(f"PPV(point)={et['precision']:.3f} | WilsonLB={et['wilson_lb']:.3f} | Recall={et['recall']:.3f} | "
          f"Coverage={et['coverage']:.3f} | CalibGap={et['calib_gap']:.3f} | FP/1k={et['fp'] / max(1,(mt[:, j].sum())) * 1000:.3f}\n")

    # FDR family (optional)
    fam = [s.strip() for s in args.safety_labels.split(",") if s.strip()]
    fdr_pass = True
    if fam and args.fdr_method != "none":
        print(f"FDR Method/Labels  : {args.fdr_method} / {','.join(fam)}")
        pvals = []
        for lab in fam:
            if lab not in labels: pvals.append(1.0); continue
            k = labels.index(lab)
            ev = evaluate_label(yv[:, k], mv[:, k], pv[:, k],
                                best_thr if k==j else load_thresholds(args.run_dir, len(labels))[k],
                                args.alpha, args.target_ppv)
            pvals.append(binom_sf_geq(ev["tp"], ev["npos"], args.target_ppv))
        keep = fdr_keep_mask(np.array(pvals, float), alpha=args.alpha, method=args.fdr_method)
        fdr_pass = bool(keep.all())
        print(f"VAL FDR Overall    : {'PASS' if fdr_pass else 'FAIL'}")
        print(f"FDR Evaluated on  : {', '.join(fam)}")

    overall = (best_val is not None) and fdr_pass
    print(f"OVERALL DECISION   : {'PASS' if overall else 'FAIL'}")

    # optional write
    if overall and args.commit:
        out = safe_update_threshold(args.run_dir, {args.primary_label: best_thr}, labels, commit=True)
        print(f"[INFO] Updated tuned_thresholds[{args.primary_label}] to {best_thr:.3f} at {out}")

if __name__ == "__main__":
    main()
