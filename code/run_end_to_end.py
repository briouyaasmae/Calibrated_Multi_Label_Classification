#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end Kaggle runner for the DepressionEmo JBI pipeline.

Steps:
  0) Train + evaluate variants (includes safety_bh variant with BH FDR)
  1) Primary threshold optimization on VAL (broad + narrow)
  2) (optional) Sweep across targets/alphas
  3) Post-hoc calibration (Platt / Isotonic), tables & plots
  4) (optional) Calibrated re-tune with one-sided binomial p-value constraint

Safe-by-default:
  - No thresholds are written unless you set commit flags below.
  - Logs are printed and also written under a 'logs/' folder.

Adjust the CONFIG block only; everything else should just work in Kaggle.
"""

import os, json, shlex, subprocess, datetime as dt
from pathlib import Path

# ============================ CONFIG ============================

CONFIG = {
    # data
    "dep_train": "/kaggle/working/Dataset_ready/train_ready.json",
    "dep_val":   "/kaggle/working/Dataset_ready/val_ready.json",
    "dep_test":  "/kaggle/working/Dataset_ready/test_ready.json",

    # where all variants go
    "all_root":  "runs/depressionemo/jbi_eval_plus",

    # backbone
    "model": "roberta-base",

    # safety & stats
    "safety_labels": "suicide_intent",
    "primary_label": "suicide_intent",
    "target_precision_baseline": 0.85,  # used in training run & 1st threshold tune
    "target_precision_stricter": 0.85,  # feasable
    "alpha_loose": 0.15,                # one-sided alpha for Wilson/FDR (looser)
    "alpha_tight": 0.10,                # one-sided alpha for Wilson/FDR (tighter)

    # threshold search grids
    "grid_broad":  "0.30:0.99:0.005",
    "grid_narrow": "0.90:0.97:0.001",

    # toggles for steps
    "do_train": True,
    "do_primary_threshold": True,
    "do_sweep": True,
    "do_calibration_platt": False,
    "do_calibration_isotonic": True,   # turn on if you want both
    "do_calibrated_retune_pconstr": True,

    # COMMIT SWITCHES (False by default; set True to write tuned_thresholds.npy)
    "commit_primary_threshold": False,        # Step 1 commit
    "commit_calibrated_retune_pconstr": True,# Step 4 commit

    # calibration CV
    "calib_cv_folds": 5,                 # <- use the stable 5-fold setup


    # retune safety constraints
    "min_recall": 0.20,
    "min_cov": 0.05,
    "max_calib_gap": 0.08,
    "pconstr_grid": "0.60:0.90:0.001",   # for p-constrained retune

    # significance bootstrap in training runner
    "sig_B": 1000,
    "sig_seed": 13,
    "clin_delta": 0.02
}

# ============================ UTILS ============================

LOG_ROOT = Path("logs")
LOG_ROOT.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd: str, log_name: str):
    """Run a shell command, tee output to a timestamped log and stdout."""
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_ROOT / f"{ts}_{log_name}.log"
    print(f"\n[RUN] {cmd}\n[LOG] {log_path}\n")
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed (exit {ret}): {cmd}")
    return str(log_path)

def j(path: str, *parts) -> str:
    return str(Path(path).joinpath(*parts))

# ============================ STEPS ============================

def step0_train_variants(cfg):
    cmd = f"""
python run_depr_jbi_eval_plus.py \
  --dep_train {cfg['dep_train']} \
  --dep_val   {cfg['dep_val']} \
  --dep_test  {cfg['dep_test']} \
  --all_root  {cfg['all_root']} \
  --model {cfg['model']} \
  --safety_labels {cfg['safety_labels']} \
  --primary_label {cfg['primary_label']} \
  --target_precision {cfg['target_precision_baseline']} \
  --alpha {cfg['alpha_loose']} \
  --sig_B {cfg['sig_B']} \
  --sig_seed {cfg['sig_seed']} \
  --clin_delta {cfg['clin_delta']}
""".strip()
    return run_cmd(cmd, "00_train_variants")

def step1_primary_threshold(cfg):
    rd = j(cfg["all_root"], "safety_bh")
    base = f"""
python optimize_primary_threshold.py \
  --run_dir {rd} \
  --primary_label {cfg['primary_label']} \
  --target_ppv {cfg['target_precision_baseline']} \
  --alpha {cfg['alpha_loose']} \
  --criterion wilson \
  --grid {cfg['grid_broad']} \
  --min_recall {cfg['min_recall']} \
  --min_cov {cfg['min_cov']} \
  --max_calib_gap {cfg['max_calib_gap']} \
  --safety_labels {cfg['safety_labels']} \
  --fdr_method bh
""".strip()

    run_cmd(base, "10_threshold_primary_broad")

    narrow = f"""
python optimize_primary_threshold.py \
  --run_dir {rd} \
  --primary_label {cfg['primary_label']} \
  --target_ppv {cfg['target_precision_stricter']} \
  --alpha {cfg['alpha_tight']} \
  --criterion wilson \
  --grid {cfg['grid_narrow']} \
  --min_recall {cfg['min_recall']} \
  --min_cov {cfg['min_cov']} \
  --max_calib_gap {cfg['max_calib_gap']} \
  --safety_labels {cfg['safety_labels']} \
  --fdr_method bh
""".strip()
    run_cmd(narrow, "11_threshold_primary_narrow")

    if cfg["commit_primary_threshold"]:
        commit_cmd = base + " --commit"
        run_cmd(commit_cmd, "12_threshold_primary_COMMIT")

def step2_sweep(cfg):
    rd = j(cfg["all_root"], "safety_bh")
    out_dir = j("sweep_results", cfg["primary_label"])
    cmd = f"""
python sweep_tightenups.py \
  --run_dir {rd} \
  --primary_label {cfg['primary_label']} \
  --out_dir {out_dir} \
  --safety_labels {cfg['safety_labels']} \
  --fdr_method bh
""".strip()
    run_cmd(cmd, "20_sweep")
    # If you want to commit the sweep winner uncomment below:
    # run_cmd(cmd + " --commit", "21_sweep_COMMIT")

def step3_calibration(cfg, method: str):
    rd = j(cfg["all_root"], "safety_bh")
    cmd = f"""
python calibrate_model.py \
  --run_dir {rd} \
  --method {method} \
  --cv_folds {cfg['calib_cv_folds']} \
  --target_precision {cfg['target_precision_stricter']} \
  --alpha {cfg['alpha_tight']} \
  --primary_label {cfg['primary_label']}
""".strip()
    run_cmd(cmd, f"30_calibration_{method}")

def step3_calibration_retune_commit(cfg, method: str):
    """Use if you want to *adopt* calibrated VAL operating points."""
    rd = j(cfg["all_root"], "safety_bh")
    cmd = f"""
python calibrate_model.py \
  --run_dir {rd} \
  --method {method} \
  --cv_folds {cfg['calib_cv_folds']} \
  --target_precision {cfg['target_precision_stricter']} \
  --alpha {cfg['alpha_tight']} \
  --primary_label {cfg['primary_label']} \
  --retune_thresholds \
  --commit_thresholds
""".strip()
    run_cmd(cmd, f"31_calibration_{method}_RETUNE_COMMIT")

def step4_calibrated_retune_pconstr(cfg):
    rd = j(cfg["all_root"], "safety_bh")
    base = f"""
python retune_calibrated_safety_pconstr.py \
  --run_dir {rd} \
  --label {cfg['primary_label']} \
  --target_ppv {cfg['target_precision_stricter']} \
  --alpha {cfg['alpha_loose']} \
  --min_recall {cfg['min_recall']} \
  --min_cov {cfg['min_cov']} \
  --max_calib_gap {cfg['max_calib_gap']} \
  --grid {cfg['pconstr_grid']}
""".strip()
    run_cmd(base, "40_pconstr_check")

    if cfg["commit_calibrated_retune_pconstr"]:
        run_cmd(base + " --commit", "41_pconstr_COMMIT")

# ============================ MAIN ============================

def main():
    cfg = CONFIG

    print("\n==== JBI End-to-End Runner (Kaggle) ====\n")
    print(json.dumps(cfg, indent=2))

    if cfg["do_train"]:
        step0_train_variants(cfg)

    if cfg["do_primary_threshold"]:
        step1_primary_threshold(cfg)

    if cfg["do_sweep"]:
        step2_sweep(cfg)

    if cfg["do_calibration_platt"]:
        step3_calibration(cfg, "platt")
        # If (and only if) you want to adopt calibrated VAL thresholds:
        # step3_calibration_retune_commit(cfg, "platt")

    if cfg["do_calibration_isotonic"]:
        step3_calibration(cfg, "isotonic")
        # step3_calibration_retune_commit(cfg, "isotonic")

    if cfg["do_calibrated_retune_pconstr"]:
        step4_calibrated_retune_pconstr(cfg)

    print("\n[OK] Pipeline finished. See logs/ and runs/ for outputs.\n")

if __name__ == "__main__":
    main()
