#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TailBoost (Best) — deployment-first multilabel trainer with statistical controls.

Key features kept:
- Label-Query head
- Optional graph smoothing (lambda>0)
- ASL (default) or BCE
- Rare-label sampler
- Precision-safe threshold tuning (grid + precision floor)
- Optional soft-F1 fine-tuning
- Head–tail truncation

JBI additions:
- FDR control (BH) on validation *with threshold-tightening* (no 0.5 fallback)
- Per-safety-label one-sided binomial tests at tuned thresholds
- Primary-label precision CIs on TEST (Wilson LB, Bootstrap CI, Beta(1,1) LB)
- (Optional) Save VAL arrays to support abstention tuning in eval

Artifacts (output_dir):
  best.ckpt / best_f1ft.ckpt
  test_logits.npy / test_labels.npy / test_mask.npy
  tuned_thresholds.npy + thresholds.json + label_order.json + summary.json
  final_metrics_test.json
  per_label_report.csv
  ops_safety_fdr.json                  # if safety_labels provided
  primary_label_ci.json                # if primary_label provided
  val_logits.npy / val_labels.npy / val_mask.npy  # if --save_val_arrays
"""

import os, json, math, random, argparse, csv, time
from typing import List, Dict, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer, AutoConfig, AutoModel,
    get_linear_schedule_with_warmup, DataCollatorWithPadding,
)

# ---- metrics ----
try:
    from sklearn.metrics import (
        f1_score, precision_recall_fscore_support, jaccard_score,
        precision_recall_curve
    )
except Exception:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"])
    from sklearn.metrics import f1_score, precision_recall_fscore_support, jaccard_score, precision_recall_curve

# ---- optional SciPy for Beta quantile / exact binom sf ----
try:
    from scipy.stats import beta as _scipy_beta
    from scipy.stats import binom as _scipy_binom
    _SCIPY_OK = True
except Exception:
    try:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scipy"])
        from scipy.stats import beta as _scipy_beta
        from scipy.stats import binom as _scipy_binom
        _SCIPY_OK = True
    except Exception:
        _SCIPY_OK = False


# --------------------- Repro ---------------------
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------- IO + Schema ---------------------
def read_json_or_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path): raise FileNotFoundError(path)
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln: rows.append(json.loads(ln))
        return rows
    data = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(data, list): return data
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list): return data["data"]
    raise ValueError("JSON must be a list or dict with 'data' list.")


def infer_schema(rows: List[dict]) -> Tuple[str, List[str]]:
    if not rows: raise ValueError("Empty dataset.")
    text_key = None
    for cand in ["text", "sentence", "content", "tweet", "post"]:
        if cand in rows[0]: text_key = cand; break
    if text_key is None:
        for r in rows:
            for cand in ["text", "sentence", "content", "tweet", "post"]:
                if cand in r: text_key = cand; break
            if text_key: break
    if text_key is None: raise ValueError("No text field (text/sentence/content/tweet/post).")
    labels = []
    for k, v in rows[0].items():
        if k == text_key: continue
        if v is None or isinstance(v, (int, float)): labels.append(k)
    if not labels: raise ValueError("No numeric/None label keys found.")
    return text_key, sorted(labels)


def extract_X_y_mask(rows: List[dict], text_key: str, label_order: List[str]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    texts, Y, M = [], [], []
    for r in rows:
        t = str(r.get(text_key, "") or "")
        if not t.strip(): continue
        vec, m = [], []
        for lab in label_order:
            v = r.get(lab, None)
            if v is None: vec.append(0.0); m.append(0.0)
            else:
                try: f = 1.0 if float(v) >= 0.5 else 0.0
                except Exception: f = 0.0
                vec.append(f); m.append(1.0)
        texts.append(t); Y.append(vec); M.append(m)
    return texts, np.array(Y, np.float32), np.array(M, np.float32)


# --------------------- Encoding ---------------------
def _encode_standard(tokenizer, text, max_length):
    return tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length,
                     padding=False, return_tensors=None)

def _encode_headtail(tokenizer, text, max_length, head_ratio=0.55):
    enc = tokenizer(text, add_special_tokens=False, truncation=False, padding=False, return_tensors=None)
    ids = enc["input_ids"]
    if isinstance(ids[0], list): ids = ids[0]
    max_no_specials = max(0, int(max_length) - 2)
    if len(ids) <= max_no_specials:
        full_ids = tokenizer.build_inputs_with_special_tokens(ids)
    else:
        head_len = max(1, int(round(max_no_specials * float(head_ratio))))
        tail_len = max_no_specials - head_len
        seg = ids[:head_len] + (ids[-tail_len:] if tail_len > 0 else [])
        full_ids = tokenizer.build_inputs_with_special_tokens(seg)
    out = {"input_ids": full_ids, "attention_mask": [1] * len(full_ids)}
    if tokenizer.model_input_names and "token_type_ids" in tokenizer.model_input_names:
        out["token_type_ids"] = [0] * len(out["input_ids"])
    return out

class MultiLabelTextDataset(Dataset):
    def __init__(self, texts, labels, mask, tokenizer, max_length=128,
                 truncation_mode="standard", head_ratio=0.55):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.tok = tokenizer
        self.max_length = int(max_length)
        self.truncation_mode = truncation_mode
        self.head_ratio = float(head_ratio)

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = _encode_headtail(self.tok, text, self.max_length, self.head_ratio) if self.truncation_mode == "headtail" else _encode_standard(self.tok, text, self.max_length)
        item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}
        item["labels"] = self.labels[idx]
        item["label_mask"] = self.mask[idx]
        return item


# --------------------- Label-Query Head ---------------------
class LabelQueryAttentionHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, attn_dim: int = 256,
                 dropout: float = 0.1, graph_lambda: float = 0.0, S: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_labels = num_labels
        self.graph_lambda = float(graph_lambda)
        self.register_buffer("S", None)
        if S is not None: self.register_buffer("S", S)
        self.queries = nn.Parameter(torch.randn(num_labels, attn_dim) * 0.02)
        self.key = nn.Linear(hidden_size, attn_dim, bias=False)
        self.value = nn.Linear(hidden_size, attn_dim, bias=False)
        self.out_weight = nn.Parameter(torch.randn(num_labels, attn_dim) * 0.02)
        self.out_bias   = nn.Parameter(torch.zeros(num_labels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        K = self.key(X); V = self.value(X); Q = self.queries
        if (self.S is not None) and (self.graph_lambda > 0.0):
            Q = (1.0 - self.graph_lambda) * Q + self.graph_lambda * (self.S @ Q)
        attn_logits = torch.einsum('ld,btd->blt', Q, K) / math.sqrt(K.size(-1))
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).expand(-1, self.num_labels, -1)
            attn_logits = attn_logits.masked_fill(mask == 0, torch.finfo(attn_logits.dtype).min)
        attn = attn_logits.softmax(-1)
        ctx  = torch.einsum('blt,btd->bld', attn, V)
        ctx = self.dropout(ctx)
        logits = (ctx * self.out_weight[None, :, :]).sum(-1) + self.out_bias[None, :]
        return logits


class RobertaMultiLabel(nn.Module):
    def __init__(self, model_name: str, num_labels: int,
                 gradient_checkpointing: bool = False,
                 attn_dim: int = 256, graph_lambda: float = 0.0, S: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=False)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        if gradient_checkpointing:
            try:
                self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.backbone.gradient_checkpointing_enable()
        H = self.config.hidden_size
        self.head = LabelQueryAttentionHead(H, num_labels, attn_dim=attn_dim,
                                            dropout=0.1, graph_lambda=graph_lambda, S=S)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        X = out.last_hidden_state
        logits = self.head(X, attention_mask=attention_mask)
        return logits


# --------------------- Losses ---------------------
def masked_bce_with_logits(logits, targets, mask):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
    return loss

def asymmetric_loss_with_logits(
    logits, targets, mask, gamma_pos: float = 1.0, gamma_neg: float = 2.0, clip: float = 0.05
):
    x_sigmoid = torch.sigmoid(logits)
    if clip > 0:
        xs_neg = torch.clamp(1.0 - x_sigmoid + clip, max=1.0)
        xs_pos = x_sigmoid
    else:
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid
    eps = 1e-8
    log_pos = torch.log(xs_pos.clamp(min=eps))
    log_neg = torch.log(xs_neg.clamp(min=eps))
    if gamma_pos > 0:
        log_pos = ((1.0 - x_sigmoid) ** gamma_pos) * log_pos
    if gamma_neg > 0:
        log_neg = (x_sigmoid ** gamma_neg) * log_neg
    loss = -(targets * log_pos + (1.0 - targets) * log_neg)
    loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
    return loss


# --------------------- Stats helpers (FDR & CIs) ---------------------
def _right_tail_binom_pvalue(tp: int, npos: int, p0: float) -> float:
    """One-sided binomial p-value P[X >= tp], X~Bin(npos,p0)."""
    if npos <= 0: return 1.0
    if _SCIPY_OK:
        return float(_scipy_binom.sf(tp - 1, npos, p0))
    # fallback (moderate n)
    from math import comb
    s = 0.0
    for k in range(tp, npos + 1):
        s += comb(npos, k) * (p0 ** k) * ((1 - p0) ** (npos - k))
    return float(min(1.0, max(0.0, s)))

def _fdr_keep_mask(pvals: np.ndarray, alpha: float, method: str) -> np.ndarray:
    """BH discoveries mask."""
    if method == "none": return np.ones_like(pvals, bool)
    p = np.asarray(pvals, float); n = p.size
    if n == 0: return np.array([], bool)
    order = np.argsort(p)
    ranked = p[order]
    if method == "bh":
        thresh = alpha * (np.arange(1, n+1) / n)
    else:
        raise ValueError("method must be none/bh")
    passed_prefix = ranked <= np.maximum.accumulate(thresh)
    k = np.where(passed_prefix)[0].max() + 1 if passed_prefix.any() else 0
    keep = np.zeros(n, bool)
    if k > 0: keep[order[:k]] = True
    return keep

def _naive_bootstrap_precision(y: np.ndarray, yhat: np.ndarray, B: int = 200, seed: int = 13):
    rng = np.random.RandomState(seed)
    n = len(y); stats = []
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        yy, pp = y[idx], yhat[idx]
        tp = int(((yy == 1) & (pp == 1)).sum()); npos = int((pp == 1).sum())
        stats.append(tp / npos if npos > 0 else 0.0)
    stats = np.asarray(stats)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    tp = int(((y == 1) & (yhat == 1)).sum()); npos = int((yhat == 1).sum())
    point = tp / npos if npos > 0 else 0.0
    return float(point), (float(lo), float(hi))

def _wilson_lower(tp: int, n: int, alpha: float = 0.10, one_sided: bool = True) -> float:
    if n <= 0 or tp <= 0: 
        return 0.0
    # For one-sided 90%: z ≈ 1.28155; for two-sided 90%: z ≈ 1.64485
    if one_sided:
        # alpha here is the one-sided tail probability
        z = {0.10: 1.2815515655446004, 0.05: 1.6448536269514722}.get(round(alpha, 2), 1.2815515655446004)
    else:
        # keep old behavior (two-sided 1-alpha)
        z = {0.10: 1.6448536269514722, 0.05: 1.959963984540054}.get(round(alpha, 2), 1.6448536269514722)
    phat = tp / n
    denom = 1 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) + (z * z) / (4 * n)) / n) ** 0.5 / denom
    return float(max(0.0, center - margin))

def _bayes_beta_lower(tp: int, n: int, alpha: float = 0.10, prior=(1.0, 1.0)) -> float:
    if n <= 0: return 0.0
    if _SCIPY_OK:
        a, b = prior
        return float(_scipy_beta.ppf(alpha, a + tp, b + n - tp))
    return _wilson_lower(tp, n, alpha=alpha)


# --------------------- Threshold tuning ---------------------
def _grid_for_rate(p: float) -> np.ndarray:
    if p < 0.01:   return np.linspace(0.02, 0.50, 49)
    if p < 0.02:   return np.linspace(0.05, 0.60, 56)
    if p < 0.10:   return np.linspace(0.10, 0.80, 71)
    return np.linspace(0.10, 0.90, 81)

def robust_tune_thresholds(
    Y_val: np.ndarray, logits_val: np.ndarray, M_val: Optional[np.ndarray],
    label_order: List[str], min_support: int = 6,
    shrink: float = 0.9, clip_low: float = 0.2, clip_high: float = 0.98,
    metric: str = "f1", fbeta: float = 1.0, min_improve: float = 0.005,
    precision_floor: float = 0.20,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    probs = 1 / (1 + np.exp(-logits_val))
    L = Y_val.shape[1]
    thresholds = np.full(L, 0.5, np.float32)
    rep: Dict[str, Dict[str, float]] = {}
    for j in range(L):
        m = M_val[:, j].astype(bool) if (M_val is not None) else np.ones(Y_val.shape[0], bool)
        support = int(m.sum())
        y = Y_val[m, j].astype(int)
        pj = probs[m, j]
        pred_base = (pj >= 0.5).astype(int)
        base = jaccard_score(y, pred_base, average="binary", zero_division=0) if metric == "jaccard" else f1_score(y, pred_base, average="binary", zero_division=0)
        best_t, best_s, best_p = 0.5, base, 0.0

        if support >= min_support and y.size > 0:
            p_rate = float(y.mean())
            for t in _grid_for_rate(p_rate):
                pred = (pj >= t).astype(int)
                if metric == "jaccard":
                    s = jaccard_score(y, pred, average="binary", zero_division=0)
                    p = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)[0]
                else:
                    p, r, s, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
                if precision_floor > 0.0 and p < precision_floor:
                    continue
                if s > best_s:
                    best_s, best_t, best_p = float(s), float(t), float(p)

            t_applied = float(np.clip(shrink * best_t + (1 - shrink) * 0.5, clip_low, clip_high))
            if (best_s - base) < min_improve:
                t_applied = 0.5
            thresholds[j] = t_applied
            rep[label_order[j]] = {"support": support, "base": float(base), "best": float(best_s),
                                   "best_t": float(best_t), "applied_t": float(t_applied), "best_precision": float(best_p)}
        else:
            thresholds[j] = 0.5
            rep[label_order[j]] = {"support": support, "base": float(base), "best": float(base), "best_t": 0.5, "applied_t": 0.5}
    return thresholds, rep


# --------------------- Metrics (mask-aware) ---------------------
def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    y_true = (y_true > 0).astype(int)
    y_pred = (y_pred > 0).astype(int)

    if mask is not None:
        mask = mask.astype(bool)

        if mask.sum() > 0:
            yt = y_true[mask].astype(int)
            yp = y_pred[mask].astype(int)
            tp = int((yt & yp).sum())
            fp = int(((1 - yt) & yp).sum())
            fn = int((yt & (1 - yp)).sum())
            micro_f1 = 0.0 if (2 * tp + fp + fn) == 0 else (2 * tp) / (2 * tp + fp + fn)
        else:
            micro_f1 = 0.0

        L = y_true.shape[1]
        per_label_f1 = []
        for j in range(L):
            mj = mask[:, j]
            if mj.any():
                per_label_f1.append(
                    f1_score(y_true[mj, j], y_pred[mj, j], average="binary", zero_division=0)
                )
        macro_f1 = float(np.mean(per_label_f1)) if per_label_f1 else 0.0

        per_sample_f1 = []
        for i in range(y_true.shape[0]):
            mi = mask[i, :]
            if mi.any():
                per_sample_f1.append(
                    f1_score(y_true[i, mi], y_pred[i, mi], average="binary", zero_division=0)
                )
        samples_f1 = float(np.mean(per_sample_f1)) if per_sample_f1 else 0.0

        per_label_j = []
        for j in range(y_true.shape[1]):
            mj = mask[:, j]
            if mj.any():
                per_label_j.append(
                    jaccard_score(y_true[mj, j], y_pred[mj, j], average="binary", zero_division=0)
                )
        jacc_macro = float(np.mean(per_label_j)) if per_label_j else 0.0

        per_sample_j = []
        for i in range(y_true.shape[0]):
            mi = mask[i, :]
            if mi.any():
                per_sample_j.append(
                    jaccard_score(y_true[i, mi], y_pred[i, mi], average="binary", zero_division=0)
                )
        jacc_samples = float(np.mean(per_sample_j)) if per_sample_j else 0.0

        exact = []
        for i in range(y_true.shape[0]):
            mi = mask[i, :]
            if mi.any():
                exact.append(bool(np.array_equal(y_true[i, mi], y_pred[i, mi])))
        subset_acc = float(np.mean(exact)) if exact else 0.0

        hamming = float(np.not_equal(y_true[mask], y_pred[mask]).mean()) if mask.sum() > 0 else 0.0

    else:
        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        samples_f1 = f1_score(y_true, y_pred, average="samples", zero_division=0)
        jacc_macro = jaccard_score(y_true, y_pred, average="macro", zero_division=0)
        jacc_samples = jaccard_score(y_true, y_pred, average="samples", zero_division=0)
        subset_acc = float((y_true == y_pred).all(axis=1).mean())
        hamming = float(np.not_equal(y_true, y_pred).mean())

    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "samples_f1": float(samples_f1),
        "jaccard_macro": float(jacc_macro),
        "jaccard_samples": float(jacc_samples),
        "subset_accuracy": float(subset_acc),
        "hamming_loss": float(hamming),
    }

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def finalize_and_report(
    y_true: np.ndarray,
    logits: np.ndarray,
    thresholds: np.ndarray,
    out_dir: str,
    split_name: str = "TEST",
    mask: Optional[np.ndarray] = None,
    extra_kv: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    os.makedirs(out_dir, exist_ok=True)
    probs = _sigmoid(logits.astype(np.float64))
    y_pred = (probs >= thresholds[None, :]).astype(int)
    scores = compute_all_metrics(y_true, y_pred, mask=mask)
    scores["micro_acc"] = float(1.0 - scores["hamming_loss"])
    if extra_kv:
        scores.update(extra_kv)
    out_json = os.path.join(out_dir, f"final_metrics_{split_name.lower()}.json")
    with open(out_json, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\n=== Final {split_name} Scores ===")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()})
    return scores


# --------------------- Args ---------------------
def build_argparser():
    p = argparse.ArgumentParser("TailBoost (Best) — lean & strong")
    # data
    p.add_argument("--train", required=True, type=str)
    p.add_argument("--val", type=str, default=None)
    p.add_argument("--test", required=True, type=str)
    p.add_argument("--output_dir", type=str, default="./runs/tailboost_best")
    # model/backbone
    p.add_argument("--model_name", type=str, default="roberta-base")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--truncation_mode", type=str, choices=["standard","headtail"], default="standard")
    p.add_argument("--head_ratio", type=float, default=0.55)
    p.add_argument("--attn_dim", type=int, default=256)
    p.add_argument("--graph_lambda", type=float, default=0.12)
    p.add_argument("--gradient_checkpointing", action="store_true")
    # train
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # loss
    p.add_argument("--loss", type=str, choices=["asl","bce"], default="asl")
    p.add_argument("--asl_gamma_pos", type=float, default=1.0)
    p.add_argument("--asl_gamma_neg", type=float, default=2.0)
    p.add_argument("--asl_clip", type=float, default=0.05)
    # soft-F1 FT
    p.add_argument("--f1_finetune_epochs", type=int, default=1)
    p.add_argument("--f1_finetune_beta", type=float, default=1.0)
    p.add_argument("--f1_finetune_lr_scale", type=float, default=0.5)
    p.add_argument("--f1_finetune_freeze_backbone", action="store_true", default=True)
    # threshold tuning
    p.add_argument("--threshold_mode", type=str, choices=["auto","none"], default="auto")
    p.add_argument("--th_metric", type=str, choices=["f1","jaccard"], default="f1")
    p.add_argument("--th_fbeta", type=float, default=1.0)
    p.add_argument("--th_min_pos_support", type=int, default=6)
    p.add_argument("--th_shrink", type=float, default=0.9)
    p.add_argument("--th_clip_low", type=float, default=0.20)
    p.add_argument("--th_clip_high", type=float, default=0.98)
    p.add_argument("--th_min_improve", type=float, default=0.005)
    p.add_argument("--th_precision_floor", type=float, default=0.20)

    # JBI — statistical controls
    p.add_argument("--safety_labels", type=str, default="", help="Comma-separated labels to FDR-control; blank = none")
    p.add_argument("--target_precision", type=float, default=0.90, help="PPV floor used for tests on VAL")
    p.add_argument("--fdr_method", type=str, choices=["none","bh"], default="bh")
    p.add_argument("--fdr_alpha", type=float, default=0.10)
    p.add_argument("--primary_label", type=str, default="", help="Primary safety label for CI comparison on TEST")
    p.add_argument("--bootstrap_B", type=int, default=1000, help="Bootstrap reps for precision CI")

    # extras
    p.add_argument("--use_rare_sampler", action="store_true", default=True)
    p.add_argument("--sampler_power", type=float, default=0.6)
    p.add_argument("--rare_label_threshold", type=float, default=0.08)
    p.add_argument("--save_pr_curves", action="store_true", default=False)
    p.add_argument("--save_calibration_bins", action="store_true", default=False)
    p.add_argument("--save_val_arrays", action="store_true", default=True, help="Save VAL logits/labels/mask for abstention tuning")
    return p


# --------------------- Utils ---------------------
def build_safe_collate_fn(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

def build_rare_sampler_weights(Y_tr: np.ndarray, rare_thresh: float, power: float):
    p = Y_tr.mean(0)
    rare = p <= rare_thresh
    base = np.ones(Y_tr.shape[0], dtype=np.float32)
    if not rare.any(): return torch.tensor(base, dtype=torch.float32)
    wj = ((p[rare] + 1e-6) ** (-power))
    wj = wj / (wj.mean() + 1e-6)
    contrib = Y_tr[:, rare] @ wj
    weights = base + contrib
    return torch.tensor(weights, dtype=torch.float32)


# --------------------- Safety ops with FDR (tightening) ---------------------
def _val_ppv_counts_for_label(Yv: np.ndarray, Pv: np.ndarray, Mv: np.ndarray, j: int, thr: float):
    m = Mv[:, j].astype(bool)
    y = Yv[m, j].astype(int)
    pred = (Pv[m, j] >= thr).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    npos = int(pred.sum())
    return tp, npos

def _tighten_thresholds_with_fdr(
    label_order: List[str],
    Yv: np.ndarray, logits_v: np.ndarray, Mv: np.ndarray,
    thresholds: np.ndarray,
    safety_labels: List[str],
    target_precision: float,
    alpha: float,
    method: str = "bh",
    max_iter: int = 200
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, bool]]:
    """Monotone-increase thresholds for safety labels until BH passes or we hit upper bound."""
    Pv = 1.0 / (1.0 + np.exp(-logits_v))
    lab2idx = {lab: i for i, lab in enumerate(label_order)}
    safelabs = [lab for lab in safety_labels if lab in lab2idx]
    if not safelabs:
        return thresholds, {}, {}

    # per-label candidate grid from current threshold upward
    grids = {}
    for lab in safelabs:
        j = lab2idx[lab]
        base = thresholds[j]
        # build monotone grid up to 0.99 (ensuring base included)
        cand = np.unique(np.concatenate([np.linspace(max(base, 0.1), 0.99, 60), [base]]))
        grids[lab] = np.sort(cand)

    # helper to compute p-values at current thresholds
    def pvals_at(th_vec):
        pvals = []
        for lab in safelabs:
            j = lab2idx[lab]
            tp, npos = _val_ppv_counts_for_label(Yv, Pv, Mv, j, float(th_vec[j]))
            pvals.append(_right_tail_binom_pvalue(tp, npos, target_precision))
        return np.array(pvals, float)

    th = thresholds.copy()
    # iterative tightening
    for _ in range(max_iter):
        pv = pvals_at(th)
        keep = _fdr_keep_mask(pv, alpha, method)
        if keep.all():  # all pass
            break
        # pick worst failing label (largest p)
        fail_idx = np.where(~keep)[0]
        worst = fail_idx[np.argmax(pv[fail_idx])]
        lab = safelabs[worst]
        j = lab2idx[lab]
        # move j to next higher candidate
        g = grids[lab]
        curr = float(th[j])
        nxts = g[g > curr + 1e-9]
        if nxts.size == 0:
            # cannot tighten further; stop
            break
        th[j] = float(nxts[0])

    # final pass/fail map and pvals
    final_p = pvals_at(th)
    final_keep = _fdr_keep_mask(final_p, alpha, method)
    pmap = {lab: float(final_p[i]) for i, lab in enumerate(safelabs)}
    kmap = {lab: bool(final_keep[i]) for i, lab in enumerate(safelabs)}
    return th, pmap, kmap


# --------------------- Train / Eval ---------------------
def run(args):
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print("\n=== CONFIG ===")
    print(json.dumps(vars(args), indent=2))

    rows_tr = read_json_or_jsonl(args.train)
    text_key, label_order = infer_schema(rows_tr)
    rows_te = read_json_or_jsonl(args.test)
    rows_va = read_json_or_jsonl(args.val) if args.val else []

    tr_texts, Y_tr, M_tr = extract_X_y_mask(rows_tr, text_key, label_order)
    te_texts, Y_te, M_te = extract_X_y_mask(rows_te, text_key, label_order)
    if rows_va:
        va_texts, Y_va, M_va = extract_X_y_mask(rows_va, text_key, label_order)
    else:
        va_texts, Y_va, M_va = [], np.zeros((0, len(label_order)), np.float32), np.zeros((0, len(label_order)), np.float32)

    print(f"\nLabels ({len(label_order)}): {label_order}")
    print(f"Train={len(tr_texts)}  Val={len(va_texts)}  Test={len(te_texts)}")

    # Tokenizer & data
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if getattr(tok, "eos_token", None) else tok.cls_token
    collate_fn = build_safe_collate_fn(tok)

    ds_tr = MultiLabelTextDataset(tr_texts, Y_tr, M_tr, tok, max_length=args.max_length,
                                  truncation_mode=args.truncation_mode, head_ratio=args.head_ratio)
    ds_te = MultiLabelTextDataset(te_texts, Y_te, M_te, tok, max_length=args.max_length,
                                  truncation_mode=args.truncation_mode, head_ratio=args.head_ratio)
    ds_va = MultiLabelTextDataset(va_texts, Y_va, M_va, tok, max_length=args.max_length,
                                  truncation_mode=args.truncation_mode, head_ratio=args.head_ratio) if len(va_texts) else None

    # Rare-label sampler
    sampler = None
    if args.use_rare_sampler:
        print("[info] enabling rare-label sampler")
        w = build_rare_sampler_weights(Y_tr, args.rare_label_threshold, args.sampler_power)
        sampler = WeightedRandomSampler(weights=w, num_samples=len(ds_tr), replacement=True)

    nworkers = max(2, (os.cpu_count() or 2) // 2)
    loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                           num_workers=nworkers, pin_memory=True, persistent_workers=True, prefetch_factor=2, collate_fn=collate_fn)
    loader_te = DataLoader(ds_te, batch_size=args.eval_batch_size, shuffle=False,
                           num_workers=nworkers, pin_memory=True, persistent_workers=True, prefetch_factor=2, collate_fn=collate_fn)
    loader_va = None
    if ds_va:
        loader_va = DataLoader(ds_va, batch_size=args.eval_batch_size, shuffle=False,
                               num_workers=nworkers, pin_memory=True, persistent_workers=True, prefetch_factor=2, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Label co-occurrence for graph smoothing
    with np.errstate(divide='ignore', invalid='ignore'):
        co = (Y_tr.T @ Y_tr).astype(np.float32) + 1e-6
        co = co / np.clip(co.sum(axis=1, keepdims=True), 1e-6, None)
    S = torch.tensor(co, dtype=torch.float32, device=device)

    model = RobertaMultiLabel(args.model_name, num_labels=Y_tr.shape[1],
                              gradient_checkpointing=args.gradient_checkpointing,
                              attn_dim=args.attn_dim,
                              graph_lambda=args.graph_lambda,
                              S=S if args.graph_lambda > 0 else None).to(device)

    # Bias init with priors
    with torch.no_grad():
        p = torch.tensor(Y_tr.mean(0), dtype=torch.float32, device=device).clamp(1e-4, 1-1e-4)
        prior_logit = torch.log(p/(1-p))
        model.head.out_bias.copy_(prior_logit)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    total_steps = math.ceil(len(loader_tr) / max(1, args.grad_accum)) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16

    best_val_macro = -1.0; best_state = None; no_improve = 0
    print("\n=== Training ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader_tr, desc=f"Epoch {epoch}", leave=False)
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar, start=1):
            labels = batch.pop("labels").to(device)
            label_mask = batch.pop("label_mask").to(device)
            for k in ("input_ids","attention_mask","token_type_ids"):
                if k in batch: batch[k] = batch[k].to(device)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
                logits = model(**batch)
                if args.loss == "asl":
                    loss = asymmetric_loss_with_logits(logits, labels, label_mask,
                                                       gamma_pos=args.asl_gamma_pos, gamma_neg=args.asl_gamma_neg, clip=args.asl_clip)
                else:
                    loss = masked_bce_with_logits(logits, labels, label_mask)
                loss = loss / max(1, args.grad_accum)
            scaler.scale(loss).backward()
            total_loss += loss.item() * max(1, args.grad_accum)
            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            pbar.set_postfix(loss=f"{total_loss/step:.4f}")

        # validate
        if loader_va is not None:
            model.eval()
            logits_va, labels_va, mask_va = [], [], []
            with torch.no_grad():
                for batch in loader_va:
                    labels = batch.pop("labels").to(device)
                    m = batch.pop("label_mask").to(device)
                    for k in ("input_ids","attention_mask","token_type_ids"):
                        if k in batch: batch[k] = batch[k].to(device)
                    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
                        logits = model(**batch)
                    logits_va.append(logits.float().cpu().numpy())
                    labels_va.append(labels.cpu().numpy())
                    mask_va.append(m.cpu().numpy())
            va_logits = np.concatenate(logits_va, 0) if logits_va else np.zeros((0, Y_tr.shape[1]), np.float32)
            Yv = np.concatenate(labels_va, 0) if labels_va else np.zeros((0, Y_tr.shape[1]), np.float32)
            Mv = np.concatenate(mask_va, 0) if mask_va else np.zeros((0, Y_tr.shape[1]), np.float32)
            va_pred = (1/(1+np.exp(-va_logits)) >= 0.5).astype(int)
            scores = compute_all_metrics(Yv, va_pred, mask=Mv)
            print(f"Epoch {epoch:>2d}  train_loss={total_loss/len(loader_tr):.4f}  VAL macroF1={scores['macro_f1']:.4f}  microF1={scores['micro_f1']:.4f}")
            if scores["macro_f1"] > best_val_macro:
                best_val_macro = scores["macro_f1"]; no_improve = 0
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save({"state_dict": best_state, "label_order": label_order, "args": vars(args)},
                           os.path.join(args.output_dir, "best.ckpt"))
            else:
                no_improve += 1
                if no_improve >= 2:
                    print("Early stopping."); break

    # load best
    ckpt = os.path.join(args.output_dir, "best.ckpt")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"]); model.to(device)

    # Optional Soft-F1 fine-tune
    if args.f1_finetune_epochs > 0:
        print("\n=== Soft-F1 fine-tuning ===")
        if args.f1_finetune_freeze_backbone:
            for p in model.backbone.parameters(): p.requires_grad = False
        ft_lr = args.learning_rate * args.f1_finetune_lr_scale
        opt_ft = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=ft_lr, weight_decay=args.weight_decay)
        steps = math.ceil(len(loader_tr) / max(1, args.grad_accum)) * args.f1_finetune_epochs
        sch_ft = get_linear_schedule_with_warmup(opt_ft, int(steps*0.1), steps)
        model.train()
        total = 0.0
        pbar = tqdm(range(steps), desc="F1-FT", leave=False)
        it = iter(loader_tr)
        for step in pbar:
            try: batch = next(it)
            except StopIteration:
                it = iter(loader_tr); batch = next(it)
            labels = batch.pop("labels").to(device)
            label_mask = batch.pop("label_mask").to(device)
            for k in ("input_ids","attention_mask","token_type_ids"):
                if k in batch: batch[k] = batch[k].to(device)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
                logits = model(**batch)
                probs = torch.sigmoid(logits) * label_mask
                targ  = labels * label_mask
                tp = (probs * targ).sum(0)
                fp = (probs * (1.0 - targ)).sum(0)
                fn = ((1.0 - probs) * targ).sum(0)
                beta2 = args.f1_finetune_beta ** 2
                soft_f1 = ((1 + beta2) * tp + 1e-8) / ((1 + beta2) * tp + beta2 * fn + fp + 1e-8)
                denom = (label_mask.sum(0) > 0).float().sum().clamp(min=1.0)
                loss = (1.0 - soft_f1).sum() / denom
                loss = loss / max(1, args.grad_accum)
            scaler.scale(loss).backward()
            total += loss.item() * max(1, args.grad_accum)
            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(opt_ft)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(opt_ft); scaler.update()
                opt_ft.zero_grad(set_to_none=True)
                sch_ft.step()
            pbar.set_postfix(loss=f"{total/(step+1):.4f}")

        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save({"state_dict": best_state, "label_order": label_order, "args": vars(args)},
                   os.path.join(args.output_dir, "best_f1ft.ckpt"))

    # Threshold tuning (VAL)
    if (loader_va is not None) and (args.threshold_mode == "auto"):
        model.eval()
        logits_va, labels_va, mask_va = [], [], []
        with torch.no_grad():
            for batch in loader_va:
                labels = batch.pop("labels").to(device)
                m = batch.pop("label_mask").to(device)
                for k in ("input_ids","attention_mask","token_type_ids"):
                    if k in batch: batch[k] = batch[k].to(device)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
                    logits = model(**batch)
                logits_va.append(logits.float().cpu().numpy())
                labels_va.append(labels.cpu().numpy())
                mask_va.append(m.cpu().numpy())
        va_logits = np.concatenate(logits_va, 0)
        Yv = np.concatenate(labels_va, 0)
        Mv = np.concatenate(mask_va, 0)

        th, th_report = robust_tune_thresholds(
            Yv, va_logits, Mv, label_order,
            min_support=args.th_min_pos_support, shrink=args.th_shrink,
            clip_low=args.th_clip_low, clip_high=args.th_clip_high,
            metric=args.th_metric, fbeta=args.th_fbeta, min_improve=args.th_min_improve,
            precision_floor=args.th_precision_floor
        )

        # FDR tightening for safety labels
        safety = [s.strip() for s in (args.safety_labels.split(",") if args.safety_labels else []) if s.strip()]
        pmap, kmap = {}, {}
        if safety and (args.fdr_method != "none"):
            th_tight, pmap, kmap = _tighten_thresholds_with_fdr(
                label_order=label_order, Yv=Yv, logits_v=va_logits, Mv=Mv,
                thresholds=th, safety_labels=safety,
                target_precision=args.target_precision, alpha=args.fdr_alpha, method=args.fdr_method
            )
            th = th_tight

            # persist ops table
            ops_rows = []
            Pv = 1.0 / (1.0 + np.exp(-va_logits))
            for lab in safety:
                j = label_order.index(lab) if lab in label_order else -1
                if j >= 0:
                    tp, npos = _val_ppv_counts_for_label(Yv, Pv, Mv, j, float(th[j]))
                    ppv_point = (tp / npos) if npos > 0 else 0.0
                else:
                    ppv_point = 0.0
                ops_rows.append({
                    "label": lab,
                    "threshold_val_after_FDR": float(th[j]) if j >= 0 else None,
                    "p_value_val": float(pmap.get(lab, float("nan"))),
                    "passed_fdr": bool(kmap.get(lab, False)),
                    "method": args.fdr_method,
                    "alpha": float(args.fdr_alpha),
                    "target_precision": float(args.target_precision),
                    "ppv_val_point": float(ppv_point)
                })
            with open(os.path.join(args.output_dir, "ops_safety_fdr.json"), "w") as f:
                json.dump(ops_rows, f, indent=2)

        np.save(os.path.join(args.output_dir, "tuned_thresholds.npy"), th)
        if args.save_val_arrays:
            np.save(os.path.join(args.output_dir, "val_logits.npy"), va_logits)
            np.save(os.path.join(args.output_dir, "val_labels.npy"), Yv)
            np.save(os.path.join(args.output_dir, "val_mask.npy"), Mv)
        thresholds = th
        print(f"Tuned thresholds for {int(np.sum(thresholds!=0.5))}/{len(label_order)} labels.")
    else:
        thresholds = np.full(len(label_order), 0.5, np.float32)
        print("No threshold tuning, using 0.5 for all.")

    # TEST inference → save arrays
    model.eval()
    logits_te, labels_te, mask_te = [], [], []
    total_tokens = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in loader_te:
            labels = batch.pop("labels").to(device)
            m = batch.pop("label_mask").to(device)
            for k in ("input_ids","attention_mask","token_type_ids"):
                if k in batch: batch[k] = batch[k].to(device)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
                logits = model(**batch)
            logits_te.append(logits.float().cpu().numpy())
            labels_te.append(labels.cpu().numpy())
            mask_te.append(m.cpu().numpy())
            total_tokens += int(batch["input_ids"].numel())
    t1 = time.time()

    te_logits = np.concatenate(logits_te, 0)
    Yt = np.concatenate(labels_te, 0)
    Mt = np.concatenate(mask_te, 0)
    np.save(os.path.join(args.output_dir, "test_logits.npy"), te_logits)
    np.save(os.path.join(args.output_dir, "test_labels.npy"), Yt)
    np.save(os.path.join(args.output_dir, "test_mask.npy"), Mt)

    # Per-label probs/preds
    probs = 1.0 / (1.0 + np.exp(-te_logits))
    Y_hat = (probs >= thresholds[None, :]).astype(int)

    # Final metrics
    scores = finalize_and_report(
        y_true=Yt,
        logits=te_logits,
        thresholds=thresholds,
        out_dir=args.output_dir,
        split_name="TEST",
        mask=Mt,
        extra_kv={"throughput_tokens_per_sec": float(total_tokens / max(1e-6, (t1 - t0)))}
    )

    # ---- Primary label precision CIs on TEST
    if args.primary_label and args.primary_label in label_order:
        j = label_order.index(args.primary_label)
        m = Mt[:, j].astype(bool)
        y = Yt[m, j].astype(int)
        pred = Y_hat[m, j].astype(int)
        tp = int(((y == 1) & (pred == 1)).sum()); npos = int(pred.sum())
        point = tp / npos if npos > 0 else 0.0
        wil_lb = _wilson_lower(tp, npos, alpha=max(1e-6, args.fdr_alpha))
        b_point, (blo, bhi) = _naive_bootstrap_precision(y, pred, B=max(200, args.bootstrap_B))
        bayes_lb = _bayes_beta_lower(tp, npos, alpha=max(1e-6, args.fdr_alpha), prior=(1.0, 1.0))
        with open(os.path.join(args.output_dir, "primary_label_ci.json"), "w") as f:
            json.dump({
                "label": args.primary_label,
                "threshold_at_test": float(thresholds[j]),
                "precision_point": float(point),
                "wilson_lower": float(wil_lb),
                "bootstrap_point": float(b_point),
                "bootstrap_ci95": [float(blo), float(bhi)],
                "bayes_beta_lower": float(bayes_lb),
                "alpha": float(args.fdr_alpha)
            }, f, indent=2)

    # per-label CSV
    from sklearn.metrics import precision_recall_fscore_support as prfs
    per_label = {}
    for j, lab in enumerate(label_order):
        m = Mt[:, j].astype(bool)
        if not m.any():
            per_label[lab] = {"support": 0, "f1": 0.0, "thr": float(thresholds[j])}
            continue
        y_true = Yt[m, j]; y_pred = Y_hat[m, j]
        p, r, f1, _ = prfs(y_true, y_pred, average="binary", zero_division=0)
        per_label[lab] = {"support": int(m.sum()), "precision": float(p), "recall": float(r), "f1": float(f1),
                          "thr": float(thresholds[j])}
    with open(os.path.join(args.output_dir, "per_label_report.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["label","support","thr","precision","recall","f1"])
        for lab in label_order:
            d = per_label[lab]
            w.writerow([lab, d["support"], d["thr"], d.get("precision",0.0), d.get("recall",0.0), d["f1"]])

    # human-friendly dumps
    with open(os.path.join(args.output_dir, "thresholds.json"), "w") as f:
        json.dump({lab: float(t) for lab, t in zip(label_order, thresholds)}, f, indent=2)
    with open(os.path.join(args.output_dir, "label_order.json"), "w") as f:
        json.dump(label_order, f, indent=2)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"scores": scores, "args": vars(args)}, f, indent=2)

    print(f"Saved to {args.output_dir}")

    # Optional exports
    if args.save_pr_curves:
        os.makedirs(os.path.join(args.output_dir, "pr_curves"), exist_ok=True)
        for j, lab in enumerate(label_order):
            m = Mt[:, j].astype(bool)
            if not m.any(): continue
            y = Yt[m, j].astype(int)
            p = probs[m, j]
            pr, rc, _ = precision_recall_curve(y, p)
            np.save(os.path.join(args.output_dir, "pr_curves", f"{lab}_precision.npy"), pr)
            np.save(os.path.join(args.output_dir, "pr_curves", f"{lab}_recall.npy"), rc)
    if args.save_calibration_bins:
        os.makedirs(os.path.join(args.output_dir, "calibration"), exist_ok=True)
        bins = np.linspace(0, 1, 11)
        mids = (bins[:-1] + bins[1:]) / 2.0
        for j, lab in enumerate(label_order):
            m = Mt[:, j].astype(bool)
            if not m.any(): continue
            y = Yt[m, j].astype(int); p = probs[m, j]
            counts = np.zeros(10, int); pos = np.zeros(10, int)
            for i in range(10):
                sel = (p >= bins[i]) & (p < bins[i+1]) if i < 9 else (p >= bins[i]) & (p <= bins[i+1])
                counts[i] = int(sel.sum()); pos[i] = int(y[sel].sum())
            with open(os.path.join(args.output_dir, "calibration", f"{lab}_calibration.json"), "w") as f:
                json.dump({"bin_mid": mids.tolist(), "count": counts.tolist(), "pos": pos.tolist()}, f, indent=2)


def main():
    p = build_argparser()
    args = p.parse_args()
    run(args)

if __name__ == "__main__":
    main()
