#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DepressionEmo JBI evaluation with deployment-focused auditing.

What it does:
- Trains 5 variants (base, no_graph, no_asl, no_f1_ft, no_headtail) + safety_bh
- Summarizes metrics (incl. calibration at OP)
- Learns an abstention band δ on VAL to keep PPV≥target with max coverage; evaluates on TEST
- Subgroup PPV@OP (if --subgroup_key present in JSON rows)
- Paired bootstrap deltas vs base with BH correction
- Li–Ji effective number of tests (supplemental)
- Weekly monitoring sample size for PPV auditing (Wilson half-width)

Outputs under --all_root:
  summary_variants.csv
  significance_vs_base.csv
  abstention_summary.csv
  subgroup_ppv.csv           (if subgroup_key provided)
  monitoring_plan.json
  li_ji_effective_tests.json
"""

import os, json, argparse, copy, sys, subprocess
from argparse import Namespace
from typing import Dict, List, Tuple
import numpy as np

# statsmodels for BH
try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "statsmodels"], check=False)
    from statsmodels.stats.multitest import multipletests

import train_tailboost_best as tbpp


# ---------- utils ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def write_csv(path: str, rows: List[Dict[str, str]], header: List[str]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in header) + "\n")
    print(f"[OK] wrote {path}")

def sigmoid(x): 
    z = np.clip(x, -50, 50); return 1.0/(1.0+np.exp(-z))

def masked_metrics_micro_macro(Y: np.ndarray, Yhat: np.ndarray, M: np.ndarray) -> Tuple[float, float]:
    Yb = (Y > 0).astype(np.int8)
    Yp = (Yhat > 0).astype(np.int8)
    Mb = M.astype(bool)
    if Mb.sum() == 0: return 0.0, 0.0
    yt, yp = Yb[Mb], Yp[Mb]
    tp = int((yt & yp).sum()); fp = int(((1 - yt) & yp).sum()); fn = int((yt & (1 - yp)).sum())
    micro = 0.0 if (2*tp + fp + fn) == 0 else (2*tp) / (2*tp + fp + fn)
    L = Y.shape[1]; per = []
    for j in range(L):
        mj = Mb[:, j]; 
        if not mj.any(): continue
        yj, pj = Yb[mj, j], Yp[mj, j]
        tpj = int((yj & pj).sum()); fpj = int(((1 - yj) & pj).sum()); fnj = int((yj & (1 - pj)).sum())
        denom = 2*tpj + fpj + fnj
        per.append(0.0 if denom == 0 else (2*tpj)/denom)
    macro = float(np.mean(per)) if per else 0.0
    return float(micro), float(macro)

def brier_micro(Y, P, M):
    Mb = M.astype(bool)
    if Mb.sum() == 0: return 0.0
    Yb = (Y > 0).astype(np.float64)[Mb]; Pb = np.clip(P, 1e-6, 1-1e-6)[Mb]
    return float(np.mean((Pb - Yb)**2))

def ece_equalwidth(Y, P, M, nbins=10):
    Mb = M.astype(bool)
    Yb = (Y > 0).astype(np.int32)[Mb]; Pb = np.clip(P, 0.0, 1.0)[Mb]
    N = Yb.size; 
    if N == 0: return 0.0
    bins = np.linspace(0,1,nbins+1); ece = 0.0
    for i in range(nbins):
        lo, hi = bins[i], bins[i+1]
        sel = (Pb >= lo) & (Pb <= hi if i==nbins-1 else Pb < hi)
        n = int(sel.sum())
        if n==0: continue
        acc = float(Yb[sel].mean()); conf = float(Pb[sel].mean())
        ece += (n / N) * abs(acc - conf)
    return float(ece)

def preds_at(P: np.ndarray, th: np.ndarray) -> np.ndarray:
    return (P >= th[None, :]).astype(np.int8)

def coverage_ppv_at(Y: np.ndarray, P: np.ndarray, M: np.ndarray, j: int, thr: float, delta: float = 0.0):
    """Return coverage and PPV when using abstention band delta for label j (positive if p>=thr+δ)."""
    Mb = M[:, j].astype(bool)
    if Mb.sum() == 0: return 0.0, 0.0, 0, 0
    y = (Y[Mb, j] > 0).astype(np.int32)
    p = P[Mb, j]
    pos_mask = (p >= (thr + delta))
    n_all = y.size; n_pos = int(pos_mask.sum())
    cov = float(n_pos / max(1, n_all))
    if n_pos == 0: 
        return cov, 0.0, 0, n_pos
    ppv = float(y[pos_mask].mean())
    tp = int(y[pos_mask].sum())
    return cov, ppv, tp, n_pos

def learn_delta_for_label(Yv, Pv, Mv, j, thr, target_ppv=0.90):
    """Grid search δ∈[0,0.3] maximizing coverage subject to PPV≥target on VAL."""
    grid = np.linspace(0.0, 0.30, 31)
    best = (0.0, -1.0)  # (delta, coverage)
    for d in grid:
        cov, ppv, _, _ = coverage_ppv_at(Yv, Pv, Mv, j, thr, d)
        if ppv >= target_ppv and cov > best[1]:
            best = (float(d), float(cov))
    return best[0], best[1]

def paired_bootstrap_delta(Y: np.ndarray, M: np.ndarray,
                           Yhat_A: np.ndarray, Yhat_B: np.ndarray,
                           B: int = 1000, seed: int = 13) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    n = Y.shape[0]; d_micro, d_macro = [], []
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        Ya, Ma = Y[idx], M[idx]
        A = Yhat_A[idx]; Bp = Yhat_B[idx]
        micA, macA = masked_metrics_micro_macro(Ya, A, Ma)
        micB, macB = masked_metrics_micro_macro(Ya, Bp, Ma)
        d_micro.append(micB - micA); d_macro.append(macB - macA)
    def summarise(v):
        v = np.asarray(v, float)
        lo, hi = np.percentile(v, [2.5, 97.5])
        mean = float(v.mean())
        p_two = float(min(2*min((v<=0).mean(), (v>=0).mean()), 1.0))
        sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
        d = float(mean/sd) if sd > 0 else 0.0
        ps = float((v > 0).mean() + 0.5*(v==0).mean())
        cliffs = float(2*ps - 1)
        return dict(mean=mean, lo=lo, hi=hi, p=p_two, d=d, ps=ps, cliffs=cliffs)
    sm = summarise(d_micro); sM = summarise(d_macro)
    return {
        "micro_mean": sm["mean"], "micro_ci_lo": sm["lo"], "micro_ci_hi": sm["hi"],
        "micro_p": sm["p"], "micro_cohen_d": sm["d"], "micro_ps": sm["ps"], "micro_cliffs_delta": sm["cliffs"],
        "macro_mean": sM["mean"], "macro_ci_lo": sM["lo"], "macro_ci_hi": sM["hi"],
        "macro_p": sM["p"], "macro_cohen_d": sM["d"], "macro_ps": sM["ps"], "macro_cliffs_delta": sM["cliffs"],
    }

def interpret_cohen_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.20: return "negligible"
    if ad < 0.50: return "small"
    if ad < 0.80: return "medium"
    return "large"

def interpret_cliffs(delta: float) -> str:
    a = abs(delta)
    if a < 0.147: return "negligible"
    if a < 0.330: return "small"
    if a < 0.474: return "medium"
    return "large"

def li_ji_effective_tests(corr: np.ndarray) -> float:
    """Li–Ji (2005) effective number of independent tests via eigenvalues."""
    vals = np.linalg.eigvalsh(corr)
    Meff = float(np.sum((vals > 1).astype(float) * (vals - 1) + 1))
    return Meff

def weekly_sample_size_for_ppv(target=0.90, half_width=0.05, conf=0.95):
    """Wilson half-width ≈ w at p≈target → approximate n_pos requirement."""
    # Solve by search since closed-form is messy
    from math import isfinite
    z = 1.959963984540054 if abs(conf-0.95) < 1e-6 else 1.6448536269514722
    def wilson_halfwidth(p, n):
        denom = 1 + (z*z)/n
        var_term = (p*(1-p) + (z*z)/(4*n)) / n
        return z * np.sqrt(var_term) / denom
    n = 10
    while n < 100000:
        hw = wilson_halfwidth(target, n)
        if isfinite(hw) and hw <= half_width:
            return int(n)
        n += 1
    return 100000


# ---------- variants ----------
def base_dep(args_cli) -> Namespace:
    return Namespace(
        train=args_cli.dep_train, val=args_cli.dep_val, test=args_cli.dep_test,
        output_dir=os.path.join(args_cli.all_root, "base"),
        model_name=args_cli.model,
        max_length=256, truncation_mode="headtail", head_ratio=0.6,
        batch_size=16, eval_batch_size=32,
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1, epochs=6,
        grad_accum=1, seed=42, gradient_checkpointing=False, max_grad_norm=1.0,
        attn_dim=256, graph_lambda=0.12,
        loss="asl", asl_gamma_pos=1.0, asl_gamma_neg=2.0, asl_clip=0.05,
        f1_finetune_epochs=1, f1_finetune_beta=1.0, f1_finetune_lr_scale=0.5, f1_finetune_freeze_backbone=True,
        threshold_mode="auto", th_metric="f1", th_fbeta=1.0,
        th_min_pos_support=6, th_shrink=0.9, th_clip_low=0.30, th_clip_high=0.98,
        th_min_improve=0.005, th_precision_floor=0.25,
        use_rare_sampler=True, sampler_power=0.6, rare_label_threshold=0.10,
        save_pr_curves=False, save_calibration_bins=False,
        safety_labels="", target_precision=args_cli.target_precision,
        fdr_method="none", fdr_alpha=args_cli.alpha,
        primary_label=args_cli.primary_label, bootstrap_B=args_cli.sig_B,
        save_val_arrays=True
    )

def add_variant(base_ns: Namespace, tag: str, **overrides) -> Namespace:
    n = copy.deepcopy(base_ns)
    for k, v in overrides.items():
        setattr(n, k, v)
    n.output_dir = os.path.join(os.path.dirname(base_ns.output_dir), tag)
    return n

def build_variants(args_cli) -> Dict[str, Namespace]:
    b = base_dep(args_cli)
    return {
        "base": b,
        "no_graph": add_variant(b, "no_graph", graph_lambda=0.0),
        "no_asl": add_variant(b, "no_asl", loss="bce", asl_gamma_pos=0.0, asl_gamma_neg=2.0, asl_clip=0.0),
        "no_f1_ft": add_variant(b, "no_f1_ft", f1_finetune_epochs=0),
        "no_headtail": add_variant(b, "no_headtail", truncation_mode="standard"),
        "safety_bh": add_variant(b, "safety_bh", safety_labels=args_cli.safety_labels, fdr_method="bh"),
    }


# ---------- run helpers ----------
def run_one_variant(ns: Namespace, skip_train: bool = False) -> bool:
    try:
        ensure_dir(ns.output_dir)
        print(f"\n=== RUN {ns.output_dir} ===")
        if not skip_train:
            tbpp.run(ns)
        need = ["test_logits.npy","test_labels.npy","test_mask.npy","final_metrics_test.json","label_order.json","tuned_thresholds.npy"]
        for n in need:
            p = os.path.join(ns.output_dir, n)
            if not os.path.exists(p):
                print(f"[warn] missing {p}")
        return True
    except Exception as e:
        print(f"[error] {ns.output_dir}: {e}")
        return False

def load_variant_arrays(out_dir: str):
    Yt = np.load(os.path.join(out_dir, "test_labels.npy"))
    Mt = np.load(os.path.join(out_dir, "test_mask.npy"))
    Pt = sigmoid(np.load(os.path.join(out_dir, "test_logits.npy")))
    th = np.load(os.path.join(out_dir, "tuned_thresholds.npy")) if os.path.exists(os.path.join(out_dir,"tuned_thresholds.npy")) else np.full(Pt.shape[1], 0.5, np.float32)
    # also VAL if present (for abstention)
    Yv = Mv = Pv = None
    for n in ["val_labels.npy","val_mask.npy","val_logits.npy"]:
        if not os.path.exists(os.path.join(out_dir, n)):
            return Yt, Mt, Pt, th, Yv, Mv, Pv
    Yv = np.load(os.path.join(out_dir, "val_labels.npy"))
    Mv = np.load(os.path.join(out_dir, "val_mask.npy"))
    Pv = sigmoid(np.load(os.path.join(out_dir, "val_logits.npy")))
    return Yt, Mt, Pt, th, Yv, Mv, Pv


# ---------- subgroup helpers ----------
def read_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(ln) for ln in f if ln.strip()]
        data = json.load(f)
        return data if isinstance(data, list) else data.get("data", [])

def infer_schema(rows):
    text_key = None
    for cand in ["text", "sentence", "content", "tweet", "post"]:
        if cand in rows[0]: text_key = cand; break
    if text_key is None:
        for r in rows:
            for cand in ["text","sentence","content","tweet","post"]:
                if cand in r: text_key = cand; break
            if text_key: break
    if text_key is None: raise ValueError("No text field")
    labels = [k for k,v in rows[0].items() if k!=text_key and (v is None or isinstance(v,(int,float)))]
    return text_key, sorted(labels)

def extract_mask_ordered(rows, text_key, label_order):
    texts, Y, M = [], [], []
    for r in rows:
        t = str(r.get(text_key, "") or "")
        if not t.strip(): continue
        y,m = [],[]
        for lab in label_order:
            v = r.get(lab, None)
            if v is None: y.append(0.0); m.append(0.0)
            else:
                try: y.append(1.0 if float(v)>=0.5 else 0.0); m.append(1.0)
                except: y.append(0.0); m.append(1.0)
        texts.append(t); Y.append(y); M.append(m)
    return texts, np.array(Y, np.float32), np.array(M, np.float32)

def subgroup_masks(rows, text_key, subgroup_key, kept_indices_len):
    """Return an array of subgroup values aligned to extract order length."""
    vals = []
    for r in rows:
        t = str(r.get(text_key, "") or "")
        if not t.strip(): continue
        vals.append(r.get(subgroup_key, None))
    if len(vals) != kept_indices_len:
        print("[warn] subgroup length mismatch; skipping subgroup analysis.")
        return None
    return np.array(vals, object)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dep_train", required=True)
    ap.add_argument("--dep_val", required=True)
    ap.add_argument("--dep_test", required=True)
    ap.add_argument("--all_root", type=str, default="runs/depressionemo/jbi_eval_plus")
    ap.add_argument("--model", type=str, default="roberta-base")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--sig_B", type=int, default=1000)
    ap.add_argument("--sig_seed", type=int, default=13)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--target_precision", type=float, default=0.90)
    ap.add_argument("--clin_delta", type=float, default=0.02)
    ap.add_argument("--primary_label", type=str, default="suicide_intent")
    ap.add_argument("--safety_labels", type=str, default="suicide_intent")
    ap.add_argument("--subgroup_key", type=str, default="", help="Optional JSON field for subgroup auditing (e.g., site or gender)")
    args = ap.parse_args()

    ensure_dir(args.all_root)
    variants = build_variants(args)

    ran = {}
    for tag, ns in variants.items():
        ran[tag] = run_one_variant(ns, skip_train=args.skip_train)

    # ----- summary (incl calibration at OP) -----
    rows = []
    for tag, ns in variants.items():
        if not ran.get(tag, False): continue
        out_dir = ns.output_dir
        fm_path = os.path.join(out_dir, "final_metrics_test.json")
        if not os.path.exists(fm_path): 
            print(f"[warn] missing {fm_path}; skip {tag}"); 
            continue
        fm = json.load(open(fm_path, "r"))
        Yt, Mt, Pt, th, Yv, Mv, Pv = load_variant_arrays(out_dir)
        brier = brier_micro(Yt, Pt, Mt); ece10 = ece_equalwidth(Yt, Pt, Mt, 10)
        # OP calibration
        Yhat = preds_at(Pt, th)
        Mb = Mt.astype(bool)
        Yb, Pb, Yhb = (Yt>0).astype(int)[Mb], np.clip(Pt,0,1)[Mb], Yhat[Mb]
        pos_mask = (Yhb==1); n_all=Yhb.size; n_pos=int(pos_mask.sum())
        op_cov = float(n_pos / max(1,n_all))
        op_prec = float(Yb[pos_mask].mean()) if n_pos>0 else 0.0
        op_mean_conf = float(Pb[pos_mask].mean()) if n_pos>0 else 0.0
        op_gap = abs(op_prec - op_mean_conf)
        op_brier_pos = float(np.mean((Pb[pos_mask] - Yb[pos_mask])**2)) if n_pos>0 else 0.0

        rows.append({
            "variant": tag,
            "micro_f1": f"{fm.get('micro_f1',0.0):.4f}",
            "macro_f1": f"{fm.get('macro_f1',0.0):.4f}",
            "micro_acc": f"{fm.get('micro_acc',0.0):.4f}",
            "subset_accuracy": f"{fm.get('subset_accuracy',0.0):.4f}",
            "hamming_loss": f"{fm.get('hamming_loss',0.0):.4f}",
            "brier_micro": f"{brier:.4f}",
            "ece_10bins": f"{ece10:.4f}",
            "op_coverage": f"{op_cov:.4f}",
            "op_precision": f"{op_prec:.4f}",
            "op_mean_conf_pos": f"{op_mean_conf:.4f}",
            "op_calib_gap": f"{op_gap:.4f}",
            "op_brier_pos": f"{op_brier_pos:.4f}",
            "throughput_tokens_per_sec": f"{fm.get('throughput_tokens_per_sec',0.0):.1f}",
        })
    write_csv(os.path.join(args.all_root, "summary_variants.csv"), rows,
              ["variant","micro_f1","macro_f1","micro_acc","subset_accuracy","hamming_loss",
               "brier_micro","ece_10bins","op_coverage","op_precision","op_mean_conf_pos","op_calib_gap","op_brier_pos",
               "throughput_tokens_per_sec"])

    # ----- significance vs base (paired bootstrap + BH) -----
    if not ran.get("base", False):
        print("[warn] base missing; skip significance.")
    else:
        Yb, Mb, Pb, thb, *_ = load_variant_arrays(variants["base"].output_dir)
        Yhat_b = preds_at(Pb, thb)
        sig_rows = []; p_micro=[]; p_macro=[]; tags=[]; stats={}
        for tag, ns in variants.items():
            if tag=="base" or not ran.get(tag, False): continue
            try:
                Y, M, P, th, *_ = load_variant_arrays(ns.output_dir)
                if Y.shape != Yb.shape: raise ValueError("shape mismatch vs base")
                Yhat_v = preds_at(P, th)
                s = paired_bootstrap_delta(Y, M, Yhat_b, Yhat_v, B=args.sig_B, seed=args.sig_seed)
                stats[tag] = s; p_micro.append(s["micro_p"]); p_macro.append(s["macro_p"]); tags.append(tag)
            except Exception as e:
                print(f"[warn] significance failed for {tag}: {e}")
        if tags:
            rej_mic, p_mic_adj, _, _ = multipletests(p_micro, alpha=args.alpha, method="fdr_bh")
            rej_mac, p_mac_adj, _, _ = multipletests(p_macro, alpha=args.alpha, method="fdr_bh")
            for i, tag in enumerate(tags):
                s = stats[tag]
                d_lab = interpret_cohen_d(s["micro_cohen_d"])
                cd_lab = interpret_cliffs(s["micro_cliffs_delta"])
                clinically_meaningful = "yes" if abs(s["micro_mean"]) >= args.clin_delta else "no"
                sig_rows.append({
                    "variant_vs_base": tag,
                    "delta_micro_f1_mean": f"{s['micro_mean']:.4f}",
                    "delta_micro_f1_ci": f"[{s['micro_ci_lo']:.4f},{s['micro_ci_hi']:.4f}]",
                    "p_micro": f"{s['micro_p']:.4f}",
                    "p_micro_BH": f"{p_mic_adj[i]:.4f}",
                    f"sig_micro_BH@{args.alpha:.2f}": "yes" if rej_mic[i] else "no",
                    "micro_cohen_d": f"{s['micro_cohen_d']:.3f}",
                    "micro_cohen_d_label": d_lab,
                    "micro_cliffs_delta": f"{s['micro_cliffs_delta']:.3f}",
                    "micro_cliffs_label": cd_lab,
                    f"clinically_meaningful(Δmicro≥{args.clin_delta:.3f})": clinically_meaningful,
                    "delta_macro_f1_mean": f"{s['macro_mean']:.4f}",
                    "delta_macro_f1_ci": f"[{s['macro_ci_lo']:.4f},{s['macro_ci_hi']:.4f}]",
                    "p_macro": f"{s['macro_p']:.4f}",
                    "p_macro_BH": f"{p_mac_adj[i]:.4f}",
                    f"sig_macro_BH@{args.alpha:.2f}": "yes" if rej_mac[i] else "no",
                    "macro_cohen_d": f"{s['macro_cohen_d']:.3f}",
                    "macro_cliffs_delta": f"{s['macro_cliffs_delta']:.3f}",
                })
            write_csv(os.path.join(args.all_root, "significance_vs_base.csv"), sig_rows,
                      ["variant_vs_base","delta_micro_f1_mean","delta_micro_f1_ci","p_micro","p_micro_BH",f"sig_micro_BH@{args.alpha:.2f}",
                       "micro_cohen_d","micro_cohen_d_label","micro_cliffs_delta","micro_cliffs_label",
                       f"clinically_meaningful(Δmicro≥{args.clin_delta:.3f})",
                       "delta_macro_f1_mean","delta_macro_f1_ci","p_macro","p_macro_BH",f"sig_macro_BH@{args.alpha:.2f}",
                       "macro_cohen_d","macro_cliffs_delta"])

    # ----- Abstention δ (focus on safety_bh) -----
    abst_rows = []
    if ran.get("safety_bh", False):
        Yt, Mt, Pt, th, Yv, Mv, Pv = load_variant_arrays(variants["safety_bh"].output_dir)
        labs = json.load(open(os.path.join(variants["safety_bh"].output_dir, "label_order.json")))
        safe_list = [s.strip() for s in args.safety_labels.split(",") if s.strip()]
        for lab in safe_list:
            if lab not in labs or Yv is None: continue
            j = labs.index(lab)
            delta, cov_val = learn_delta_for_label(Yv, Pv, Mv, j, float(th[j]), target_ppv=args.target_precision)
            # apply to TEST
            cov_test, ppv_test, tp, npos = coverage_ppv_at(Yt, Pt, Mt, j, float(th[j]), delta)
            abst_rows.append({
                "variant": "safety_bh",
                "label": lab,
                "threshold_tau": f"{float(th[j]):.3f}",
                "delta_selected": f"{delta:.3f}",
                "VAL_coverage_at_OP_with_delta": f"{cov_val:.4f}",
                "TEST_coverage_at_OP_with_delta": f"{cov_test:.4f}",
                "TEST_PPV_at_OP_with_delta": f"{ppv_test:.4f}",
                "TEST_tp": tp,
                "TEST_pred_pos": npos
            })
    if abst_rows:
        write_csv(os.path.join(args.all_root, "abstention_summary.csv"), abst_rows,
                  ["variant","label","threshold_tau","delta_selected","VAL_coverage_at_OP_with_delta",
                   "TEST_coverage_at_OP_with_delta","TEST_PPV_at_OP_with_delta","TEST_tp","TEST_pred_pos"])

    # ----- Subgroup PPV@OP (optional) -----
    if args.subgroup_key:
        try:
            labs = json.load(open(os.path.join(variants["safety_bh"].output_dir, "label_order.json")))
            Yt, Mt, Pt, th, *_ = load_variant_arrays(variants["safety_bh"].output_dir)
            rows_test = read_rows(args.dep_test)
            text_key, label_order = infer_schema(rows_test)
            texts, Y_re, M_re = extract_mask_ordered(rows_test, text_key, label_order)
            if Y_re.shape[0] != Yt.shape[0]:
                raise RuntimeError("order mismatch; subgroup analysis skipped.")
            subvals = subgroup_masks(rows_test, text_key, args.subgroup_key, Yt.shape[0])
            if subvals is None: 
                raise RuntimeError("subgroup mismatch")
            sg_rows = []
            safe_list = [s.strip() for s in args.safety_labels.split(",") if s.strip()]
            for lab in safe_list:
                if lab not in labs: continue
                j = labs.index(lab)
                Mb = Mt[:, j].astype(bool)
                y = (Yt[Mb, j] > 0).astype(int)
                p = Pt[Mb, j]
                pred = (p >= float(th[j])).astype(int)
                sv = subvals[Mb]
                for val in np.unique(sv):
                    mask = (sv == val)
                    if int(mask.sum()) == 0:
                        continue
                    yv = y[mask]; pv = pred[mask]
                    npos = int(pv.sum()); ppv = float((yv[pv==1].mean()) if npos>0 else 0.0)
                    sg_rows.append({
                        "label": lab, "subgroup": args.subgroup_key, "value": str(val),
                        "support_pairs": int(mask.sum()),
                        "pred_pos": npos,
                        "PPV_at_OP": f"{ppv:.4f}"
                    })
            if sg_rows:
                write_csv(os.path.join(args.all_root, "subgroup_ppv.csv"),
                          sg_rows, ["label","subgroup","value","support_pairs","pred_pos","PPV_at_OP"])
        except Exception as e:
            print(f"[warn] subgroup analysis skipped: {e}")

    # ----- Monitoring plan & Li–Ji -----
    mon = {
        "target_precision": args.target_precision,
        "confidence": 0.95,
        "half_width": 0.05,
        "required_predicted_positives_per_week": weekly_sample_size_for_ppv(target=args.target_precision, half_width=0.05, conf=0.95),
        "drift_trigger": "If Wilson lower bound for PPV at OP < target_precision, re-tune thresholds and/or increase delta; investigate data shift.",
    }
    with open(os.path.join(args.all_root, "monitoring_plan.json"), "w") as f:
        json.dump(mon, f, indent=2)
    print("[OK] wrote monitoring_plan.json")

    # Li–Ji effective tests (using train label correlations)
    rows_train = read_rows(args.dep_train)
    text_key, labs = infer_schema(rows_train)
    _, Ytr, _ = extract_mask_ordered(rows_train, text_key, labs)
    p = np.clip(Ytr.mean(0), 1e-6, 1-1e-6)
    Z = (Ytr - p) / np.sqrt(p*(1-p))
    corr = np.corrcoef(Z.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    Meff = li_ji_effective_tests(corr)
    with open(os.path.join(args.all_root, "li_ji_effective_tests.json"), "w") as f:
        json.dump({"Meff": float(Meff), "L": int(len(labs))}, f, indent=2)
    print(f"[OK] wrote li_ji_effective_tests.json (Meff≈{Meff:.2f})")


if __name__ == "__main__":
    main()
