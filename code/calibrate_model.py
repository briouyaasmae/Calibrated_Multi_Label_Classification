#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Post-hoc calibration for multi-label mental health text classification.

- Per-label calibration with CV (Platt or isotonic)
- Optional re-tuning of thresholds on calibrated VAL probabilities
- Safety evaluation with Wilson lower bounds at fixed operating points
- Saves calibrated probs, CSV summaries, and reliability plots

Usage:
  python calibrate_model.py \
      --run_dir runs/depressionemo/jbi_eval_plus/safety_bh \
      --method platt \
      --cv_folds 3 \
      --target_precision 0.85 \
      --alpha 0.15 \
      --primary_label suicide_intent \
      --retune_thresholds
"""

import os
import json
import argparse
import csv
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

warnings.filterwarnings("ignore")


# ============================= IO & basics =============================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def load_arrays(run_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.load(os.path.join(run_dir, f"{split}_labels.npy"))
    mask = np.load(os.path.join(run_dir, f"{split}_mask.npy"))
    logits = np.load(os.path.join(run_dir, f"{split}_logits.npy"))
    probs = sigmoid(logits.astype(np.float64))
    return labels, mask, probs


def load_metadata(run_dir: str) -> Tuple[List[str], np.ndarray]:
    with open(os.path.join(run_dir, "label_order.json"), "r") as f:
        label_order = json.load(f)
    th_path = os.path.join(run_dir, "tuned_thresholds.npy")
    thresholds = np.load(th_path) if os.path.exists(th_path) else np.full(len(label_order), 0.5, dtype=np.float32)
    return label_order, thresholds


def wilson_lower_bound(tp: int, n: int, alpha: float = 0.10) -> float:
    if n <= 0 or tp <= 0:
        return 0.0
    z_table = {
        0.15: 1.03643338949,
        0.10: 1.28155156554,
        0.08: 1.40507156031,
        0.05: 1.64485362695,
        0.025: 1.95996398454,
    }
    z = z_table.get(round(alpha, 3), 1.28155156554)
    p_hat = tp / n
    denom = 1.0 + (z * z) / n
    center = (p_hat + (z * z) / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + (z * z) / (4 * n)) / n) / denom
    return float(max(0.0, center - margin))


# ============================= Calibrators =============================
class PlattScaler:
    def __init__(self):
        self.clf = LogisticRegression(solver="liblinear", random_state=42)
        self.fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray):
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(probs / (1 - probs))
        self.clf.fit(logits.reshape(-1, 1), labels.astype(int))
        self.fitted = True
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("PlattScaler not fitted")
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(probs / (1 - probs))
        return self.clf.predict_proba(logits.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray):
        self.iso.fit(probs, labels.astype(int))
        self.fitted = True
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("IsotonicCalibrator not fitted")
        return self.iso.predict(probs)


# ============================= Calibration (CV OOF) =============================
def calibrate_per_label_cv(probs: np.ndarray,
                           labels: np.ndarray,
                           mask: np.ndarray,
                           method: str = "platt",
                           cv_folds: int = 3,
                           random_state: int = 42) -> np.ndarray:
    n_samples, n_labels = probs.shape
    calibrated = probs.copy()
    print(f"Calibrating {n_labels} labels using {method} with {cv_folds}-fold CV...")

    for j in range(n_labels):
        valid = mask[:, j].astype(bool)
        if valid.sum() < cv_folds * 10:
            print(f"  Label {j}: insufficient data, skip")
            continue

        p = probs[valid, j]
        y = labels[valid, j].astype(int)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cal = np.zeros_like(p)

        for tr, va in kf.split(p):
            if method == "platt":
                calr = PlattScaler().fit(p[tr], y[tr])
            elif method == "isotonic":
                calr = IsotonicCalibrator().fit(p[tr], y[tr])
            else:
                raise ValueError(f"Unknown method: {method}")
            cal[va] = calr.predict_proba(p[va])

        calibrated[valid, j] = cal

        # diagnostics
        b0 = brier_score_loss(y, p)
        b1 = brier_score_loss(y, cal)
        print(f"  Label {j}: Brier {b0:.4f} -> {b1:.4f} (Δ={b1 - b0:+.4f})")

    return calibrated


# ============================= Threshold re-tune on calibrated VAL =============================
def retune_thresholds_on_val(y_true: np.ndarray,
                             y_prob: np.ndarray,
                             mask: np.ndarray,
                             base_thresh: np.ndarray,
                             safety_only: Optional[List[int]] = None) -> np.ndarray:
    th = base_thresh.copy()
    idxs = safety_only if safety_only is not None else list(range(y_true.shape[1]))

    for j in idxs:
        mj = mask[:, j].astype(bool)
        if mj.sum() < 6:
            continue
        y = y_true[mj, j].astype(int)
        p = y_prob[mj, j]
        grid = np.unique(np.concatenate([np.linspace(0.10, 0.90, 81), np.linspace(0.90, 0.99, 91)]))
        best_f1, best_t = -1.0, th[j]
        for t in grid:
            pred = (p >= t).astype(int)
            tp = int(((y == 1) & (pred == 1)).sum())
            fp = int(((y == 0) & (pred == 1)).sum())
            fn = int(((y == 1) & (pred == 0)).sum())
            denom = 2 * tp + fp + fn
            f1 = 0.0 if denom == 0 else (2 * tp) / denom
            if f1 > best_f1 or (f1 == best_f1 and t > best_t):
                best_f1, best_t = f1, float(t)
        th[j] = best_t
    return th


# ============================= Calibration evaluation =============================
def reliability_diagram(y_true: np.ndarray,
                        y_prob: np.ndarray,
                        n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, float]:
    bins = np.linspace(0, 1, n_bins + 1)
    lowers, uppers = bins[:-1], bins[1:]
    acc, conf, counts = [], [], []
    for lo, hi in zip(lowers, uppers):
        sel = (y_prob > lo) & (y_prob <= hi)
        n = int(sel.sum())
        if n > 0:
            acc.append(float(y_true[sel].mean()))
            conf.append(float(y_prob[sel].mean()))
            counts.append(n)
        else:
            acc.append(0.0); conf.append(0.0); counts.append(0)
    acc = np.array(acc); conf = np.array(conf); counts = np.array(counts)
    ece = float(np.sum(counts * np.abs(acc - conf)) / max(1, y_true.size))
    return bins, acc, ece


def evaluate_calibration(y_true: np.ndarray,
                         y_prob_orig: np.ndarray,
                         y_prob_cal: np.ndarray,
                         mask: np.ndarray,
                         label_names: List[str]) -> Dict:
    results = {}
    for i, name in enumerate(label_names):
        valid = mask[:, i].astype(bool)
        if valid.sum() == 0:
            continue
        yt = y_true[valid, i].astype(int)
        po = np.clip(y_prob_orig[valid, i], 1e-7, 1 - 1e-7)
        pc = np.clip(y_prob_cal[valid, i], 1e-7, 1 - 1e-7)
        b0 = brier_score_loss(yt, po); b1 = brier_score_loss(yt, pc)
        l0 = log_loss(yt, po);        l1 = log_loss(yt, pc)
        _, _, e0 = reliability_diagram(yt, po); _, _, e1 = reliability_diagram(yt, pc)
        results[name] = {
            "brier_original": b0, "brier_calibrated": b1, "brier_improvement": b0 - b1,
            "logloss_original": l0, "logloss_calibrated": l1, "logloss_improvement": l0 - l1,
            "ece_original": e0, "ece_calibrated": e1, "ece_improvement": e0 - e1,
            "support": int(valid.sum())
        }
    return results


# ============================= Safety metrics =============================
def evaluate_safety_metrics(y_true: np.ndarray,
                            y_prob: np.ndarray,
                            mask: np.ndarray,
                            thresholds: np.ndarray,
                            target_precision: float,
                            alpha: float = 0.10) -> Dict:
    out = {}
    L = y_true.shape[1]
    for j in range(L):
        valid = mask[:, j].astype(bool)
        if valid.sum() == 0:
            continue
        y = y_true[valid, j].astype(int)
        p = y_prob[valid, j]
        t = thresholds[j]
        pred = (p >= t).astype(int)
        tp = int(((y == 1) & (pred == 1)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        npos = tp + fp
        precision = tp / npos if npos > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        coverage = npos / max(1, y.size)
        wlb = wilson_lower_bound(tp, npos, alpha=alpha)
        if npos > 0:
            mean_conf_pos = float(p[pred == 1].mean())
            calib_gap = abs(precision - mean_conf_pos)
        else:
            mean_conf_pos, calib_gap = 0.0, 0.0
        out[j] = {
            "threshold": float(t),
            "precision": float(precision),
            "recall": float(recall),
            "coverage": float(coverage),
            "wilson_lower_bound": float(wlb),
            "mean_confidence_positive": float(mean_conf_pos),
            "calibration_gap": float(calib_gap),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "meets_safety_target": bool(precision >= target_precision and wlb >= target_precision),
        }
    return out


# ============================= Plots =============================
def plot_calibration_curves(y_true: np.ndarray,
                            y_prob_orig: np.ndarray,
                            y_prob_cal: np.ndarray,
                            mask: np.ndarray,
                            label_names: List[str],
                            output_dir: str,
                            n_bins: int = 10):
    out_dir = os.path.join(output_dir, "calibration_plots")
    ensure_dir(out_dir)
    for i, name in enumerate(label_names):
        valid = mask[:, i].astype(bool)
        if valid.sum() < 50:
            continue
        yt = y_true[valid, i].astype(int)
        po = y_prob_orig[valid, i]
        pc = y_prob_cal[valid, i]
        bins_o, acc_o, ece_o = reliability_diagram(yt, po, n_bins)
        bins_c, acc_c, ece_c = reliability_diagram(yt, pc, n_bins)
        centers = (bins_o[:-1] + bins_o[1:]) / 2

        plt.figure(figsize=(6, 5))
        plt.plot(centers, acc_o, "o-", label=f"Original (ECE={ece_o:.3f})")
        plt.plot(centers, acc_c, "o-", label=f"Calibrated (ECE={ece_c:.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title(f"{name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_calibration.png"), dpi=150, bbox_inches="tight")
        plt.close()


# ============================= Main =============================
def main():
    ps = argparse.ArgumentParser(description="Post-hoc calibration for multi-label models")
    ps.add_argument("--run_dir", required=True, help="Directory with arrays and metadata")
    ps.add_argument("--method", choices=["platt", "isotonic"], default="platt")
    ps.add_argument("--cv_folds", type=int, default=3)
    ps.add_argument("--target_precision", type=float, default=0.90)
    ps.add_argument("--alpha", type=float, default=0.10, help="One-sided alpha for Wilson bounds")
    ps.add_argument("--primary_label", type=str, default="suicide_intent")
    ps.add_argument("--output_suffix", type=str, default="")
    ps.add_argument("--retune_thresholds", action="store_true",
                    help="Re-tune per-label thresholds on calibrated VAL probabilities")
    ps.add_argument("--retune_safety_only", action="store_true",
                    help="If set, only re-tune thresholds for labels present in safety list file (optional)")
    ps.add_argument("--safety_list_file", type=str, default="",
                    help="Path to a JSON list of safety labels; used if --retune_safety_only")
    args = ps.parse_args()

    print("=== Post-hoc Calibration Analysis ===")
    print(json.dumps({
        "run_dir": args.run_dir,
        "method": args.method,
        "cv_folds": args.cv_folds,
        "target_precision": args.target_precision,
        "alpha": args.alpha,
        "retune_thresholds": bool(args.retune_thresholds),
        "retune_safety_only": bool(args.retune_safety_only),
        "safety_list_file": args.safety_list_file or None
    }, indent=2))

    # Load data
    try:
        y_val, m_val, p_val = load_arrays(args.run_dir, "val")
        y_test, m_test, p_test = load_arrays(args.run_dir, "test")
        label_names, thresholds = load_metadata(args.run_dir)
        print(f"Loaded: {len(label_names)} labels, VAL={y_val.shape[0]}, TEST={y_test.shape[0]}")
    except FileNotFoundError as e:
        print(f"[Error] missing arrays: {e}")
        return

    # Calibrate on VAL with CV-OOF
    p_val_cal = calibrate_per_label_cv(
        probs=p_val, labels=y_val, mask=m_val,
        method=args.method, cv_folds=args.cv_folds
    )

    # Fit final calibrators on full VAL and apply to TEST
    p_test_cal = p_test.copy()
    for j in range(len(label_names)):
        valid = m_val[:, j].astype(bool)
        if valid.sum() < max(30, args.cv_folds * 10):
            continue
        if args.method == "platt":
            calr = PlattScaler().fit(p_val[valid, j], y_val[valid, j])
        else:
            calr = IsotonicCalibrator().fit(p_val[valid, j], y_val[valid, j])
        v_test = m_test[:, j].astype(bool)
        if v_test.sum() > 0:
            p_test_cal[v_test, j] = calr.predict_proba(p_test[v_test, j])

    # Optional: re-tune thresholds on calibrated VAL probabilities
    th_used = thresholds
    if args.retune_thresholds:
        safety_idxs = None
        if args.retune_safety_only and args.safety_list_file and os.path.exists(args.safety_list_file):
            safety_labels = json.load(open(args.safety_list_file, "r"))
            name2idx = {n: i for i, n in enumerate(label_names)}
            safety_idxs = [name2idx[s] for s in safety_labels if s in name2idx]
        th_used = retune_thresholds_on_val(y_val, p_val_cal, m_val, thresholds, safety_only=safety_idxs)

    # Calibration quality (VAL and TEST)
    print("\nEvaluating calibration quality...")
    val_cal = evaluate_calibration(y_val, p_val, p_val_cal, m_val, label_names)
    test_cal = evaluate_calibration(y_test, p_test, p_test_cal, m_test, label_names)

    # Safety evaluation at fixed operating point
    print("\nEvaluating safety metrics...")
    safety_val = evaluate_safety_metrics(y_val, p_val_cal, m_val, th_used, args.target_precision, args.alpha)
    safety_tst = evaluate_safety_metrics(y_test, p_test_cal, m_test, th_used, args.target_precision, args.alpha)

    # Save tables
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    cal_rows = []
    for name in label_names:
        if name in val_cal:
            vr = val_cal[name]
            tr = test_cal.get(name, {})
            cal_rows.append({
                "label": name,
                "method": args.method,
                "val_brier_improvement": f"{vr['brier_improvement']:+.4f}",
                "val_ece_improvement": f"{vr['ece_improvement']:+.4f}",
                "test_brier_improvement": f"{tr.get('brier_improvement', 0.0):+.4f}",
                "test_ece_improvement": f"{tr.get('ece_improvement', 0.0):+.4f}",
                "val_support": vr["support"],
                "test_support": tr.get("support", 0),
            })
    cal_csv = os.path.join(args.run_dir, f"calibration_evaluation{suffix}.csv")
    with open(cal_csv, "w", newline="") as f:
        if cal_rows:
            w = csv.DictWriter(f, fieldnames=cal_rows[0].keys())
            w.writeheader(); w.writerows(cal_rows)
    print(f"[OK] wrote {cal_csv}")

    safety_rows = []
    for idx, name in enumerate(label_names):
        if idx in safety_val and idx in safety_tst:
            sv, st = safety_val[idx], safety_tst[idx]
            safety_rows.append({
                "label": name,
                "threshold": f"{sv['threshold']:.3f}",
                "val_precision": f"{sv['precision']:.3f}",
                "val_wilson_lb": f"{sv['wilson_lower_bound']:.3f}",
                "val_calib_gap": f"{sv['calibration_gap']:.3f}",
                "val_meets_target": sv["meets_safety_target"],
                "test_precision": f"{st['precision']:.3f}",
                "test_wilson_lb": f"{st['wilson_lower_bound']:.3f}",
                "test_calib_gap": f"{st['calibration_gap']:.3f}",
                "test_meets_target": st["meets_safety_target"],
            })
    safety_csv = os.path.join(args.run_dir, f"safety_calibrated{suffix}.csv")
    with open(safety_csv, "w", newline="") as f:
        if safety_rows:
            w = csv.DictWriter(f, fieldnames=safety_rows[0].keys())
            w.writeheader(); w.writerows(safety_rows)
    print(f"[OK] wrote {safety_csv}")

    # Save calibrated probabilities
    np.save(os.path.join(args.run_dir, f"val_probs_calibrated{suffix}.npy"), p_val_cal)
    np.save(os.path.join(args.run_dir, f"test_probs_calibrated{suffix}.npy"), p_test_cal)
    print("[OK] saved calibrated probabilities")

    # Plots
    print("\nGenerating calibration plots...")
    plot_calibration_curves(y_val, p_val, p_val_cal, m_val, label_names, args.run_dir)

    # Primary label summary
    if args.primary_label in label_names:
        j = label_names.index(args.primary_label)
        if j in safety_tst:
            st = safety_tst[j]
            print(f"\n=== {args.primary_label.upper()} SUMMARY (Calibrated) ===")
            print(f"TEST Precision: {st['precision']:.3f}")
            print(f"TEST Wilson LB: {st['wilson_lower_bound']:.3f}")
            print(f"TEST Calibration Gap: {st['calibration_gap']:.3f}")
            print(f"Meets Safety Target: {st['meets_safety_target']}")
            print(f"Coverage: {st['coverage']:.3f}")
            print(f"Recall: {st['recall']:.3f}")

    print("\n=== Calibration Complete ===")
    print(f"Outputs saved under: {args.run_dir}")


if __name__ == "__main__":
    main()
