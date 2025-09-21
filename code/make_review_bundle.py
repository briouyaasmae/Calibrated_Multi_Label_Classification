#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_review_bundle.py
---------------------
Create a clean review bundle (code + results + figures + tables) suitable for
upload to GitHub or sharing with reviewers.

Additions:
- --include_logs : copy logs/*.log into bundle
- --result_globs : include extra wildcard paths from results variant (files/dirs)
"""

import argparse
import json
import os
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
import glob

# ------------------------------- Helpers ----------------------------------

REQS_DEFAULT = [
    "python>=3.9",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "seaborn",
    "tqdm",
    "statsmodels",
    "torch",
    "transformers",
]

RESULT_FILES_DEFAULT = [
    # metrics + summaries
    "final_metrics_test.json",
    "summary.json",
    "thresholds.json",
    "label_order.json",
    "per_label_report.csv",
    "ops_safety_fdr.json",
    "primary_label_ci.json",
    "calibration_evaluation.csv",
    "safety_calibrated.csv",
    "summary_variants.csv",
    "significance_vs_base.csv",
    "abstention_summary.csv",
    "monitoring_plan.json",
    "li_ji_effective_tests.json",
    # arrays (npy can be heavy; include by default for reproducible figures)
    "test_logits.npy",
    "test_labels.npy",
    "test_mask.npy",
    "val_logits.npy",
    "val_labels.npy",
    "val_mask.npy",
    "tuned_thresholds.npy",
    "test_probs_calibrated.npy",
    "val_probs_calibrated.npy",
]

FIG_EXTS = (".png", ".pdf")
TABLE_EXTS = (".csv", ".tex")

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_if_exists(src: Path, dst: Path, manifest: list):
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    manifest.append({
        "src": str(src),
        "dst": str(dst),
        "size": src.stat().st_size,
        "sha256": sha256_of_file(dst),
    })
    return True

def collect_files_with_names(root: Path, names: list) -> list:
    """Return list of existing files under root that match names exactly (non-recursive priority then recursive fallback)."""
    found = []
    for name in names:
        # direct hit
        direct = root / name
        if direct.exists():
            found.append(direct)
            continue
        # recursive search
        for p in root.rglob(name):
            if p.is_file():
                found.append(p)
                break
    return found

def write_text(path: Path, content: str):
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")

def make_tree(root: Path) -> str:
    lines = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        depth = len(rel.parents) - 1
        indent = "  " * depth
        lines.append(f"{indent}{rel.name}{'/' if p.is_dir() else ''}")
    return "\n".join(lines)

# ------------------------------- Main ----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_name", required=True, help="Bundle directory and archive prefix")
    ap.add_argument("--out_dir", default=".", help="Where to write the bundle (default: current dir)")
    ap.add_argument("--code_roots", nargs="+", default=["."], help="One or more roots to search for code files")
    ap.add_argument("--include_code", nargs="*", default=[], help="Explicit code files to include (searched under code_roots)")
    ap.add_argument("--extra_files", nargs="*", default=[], help="Additional files to include (e.g., LICENSE, CITATION.cff)")
    ap.add_argument("--results_root", default="", help="Root containing experiment runs (e.g., runs/depressionemo/jbi_eval_plus)")
    ap.add_argument("--results_variant", default="safety_bh", help="Which variant folder under results_root to collect (default: safety_bh)")
    ap.add_argument("--figs_root", nargs="*", default=[], help="Folders that contain paper figures/tables (e.g., jbi_paper_figures_tables)")
    ap.add_argument("--requirements", nargs="*", default=REQS_DEFAULT, help="Requirements to write (override to customize)")
    ap.add_argument("--zip_name", default="", help="Custom zip filename (default uses project_name)")
    ap.add_argument("--make_zip", action="store_true", help="Also create a .zip archive of the bundle")
    ap.add_argument("--include_logs", action="store_true", help="Include logs/*.log in the bundle")
    ap.add_argument("--result_globs", nargs="*", default=[], help="Extra glob patterns under results_variant to include (files or whole folders)")
    args = ap.parse_args()

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

    out_root = Path(args.out_dir).resolve() / f"review_bundle_{args.project_name}"
    code_dst = out_root / "code"
    results_dst = out_root / "results" / args.results_variant
    figs_dst = out_root / "figures"
    tables_dst = out_root / "paper_tables"

    if out_root.exists():
        print(f"[info] Removing existing: {out_root}")
        shutil.rmtree(out_root)
    ensure_dir(out_root)

    manifest = []

    # --- collect code ---
    ensure_dir(code_dst)
    code_found = set()
    for root in args.code_roots:
        rootp = Path(root).resolve()
        for f in collect_files_with_names(rootp, args.include_code):
            if f in code_found:
                continue
            dst = code_dst / f.name
            copy_if_exists(f, dst, manifest)
            code_found.add(f)

    # --- results ---
    if args.results_root:
        res_root = Path(args.results_root).resolve()
        variant_path = res_root / args.results_variant
        if not variant_path.exists():
            print(f"[warn] results variant not found: {variant_path}")
        else:
            ensure_dir(results_dst)
            # Copy key files
            for name in RESULT_FILES_DEFAULT:
                copy_if_exists(variant_path / name, results_dst / name, manifest)
            # Copy common subfolders if present
            for sub in ["pr_curves", "calibration", "calibration_plots"]:
                subdir = variant_path / sub
                if subdir.exists():
                    for p in subdir.rglob("*"):
                        if p.is_file():
                            dst = results_dst / sub / p.relative_to(subdir)
                            copy_if_exists(p, dst, manifest)
            # Extra globs
            for pattern in args.result_globs:
                for match in glob.glob(str(variant_path / pattern), recursive=True):
                    mp = Path(match)
                    if mp.is_dir():
                        # copy entire dir tree
                        for p in mp.rglob("*"):
                            if p.is_file():
                                dst = results_dst / p.relative_to(variant_path)
                                copy_if_exists(p, dst, manifest)
                    elif mp.is_file():
                        dst = results_dst / mp.relative_to(variant_path)
                        copy_if_exists(mp, dst, manifest)

    # --- figures + tables ---
    for root in args.figs_root:
        r = Path(root).resolve()
        if not r.exists():
            print(f"[warn] figs_root not found: {r}")
            continue
        for p in r.rglob("*"):
            if p.is_file():
                if p.suffix.lower() in FIG_EXTS:
                    dst = figs_dst / p.name
                    copy_if_exists(p, dst, manifest)
                elif p.suffix.lower() in TABLE_EXTS:
                    dst = tables_dst / p.name
                    copy_if_exists(p, dst, manifest)

    # --- logs ---
    if args.include_logs and Path("logs").exists():
        for p in Path("logs").glob("*.log"):
            dst = out_root / "logs" / p.name
            copy_if_exists(p, dst, manifest)

    # --- extras ---
    for x in args.extra_files:
        p = Path(x).resolve()
        if p.exists() and p.is_file():
            dst = out_root / p.name
            copy_if_exists(p, dst, manifest)
        else:
            print(f"[warn] extra file missing: {x}")

    # --- requirements ---
    req_txt = "\n".join(args.requirements) + "\n"
    (out_root / "requirements.txt").write_text(req_txt, encoding="utf-8")

    # --- README ---
    readme = f"""# {args.project_name} — Review Bundle


This archive contains the *exact* code and artifacts referenced in the paper.
It was generated on **{ts}** by `make_review_bundle.py`.

## Layout

```
{make_tree(out_root)}
```

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional: regenerate paper figures/tables (if script provided)
# python code/jbi_paper_figures_tables.py --figure2_mode real --make_zip
```

## Notes
- `results/` includes key metrics JSON/CSV and required `.npy` arrays to reproduce figures.
- `MANIFEST.json` lists SHA-256 checksums for every file for integrity verification.
- If any arrays are omitted to reduce size, the figure generator will fall back gracefully.
"""
    write_text(out_root / "README.md", readme)

    # --- MANIFEST ---
    manifest_path = out_root / "MANIFEST.json"
    write_text(manifest_path, json.dumps({
        "project_name": args.project_name,
        "created_at_utc": ts,
        "files": manifest,
    }, indent=2))

    # --- .gitignore (minimal) ---
    gitignore = """__pycache__/
*.pyc
*.pyo
*.DS_Store
.venv/
"""
    write_text(out_root / ".gitignore", gitignore)

    # --- zip ---
    if args.make_zip:
        zip_name = args.zip_name or f"{out_root.name}.zip"
        zip_path = out_root.parent / zip_name
        if zip_path.exists():
            zip_path.unlink()
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=out_root)
        print(f"[ok] Bundle zipped: {zip_path}")
    print(f"[ok] Bundle created at: {out_root}")
    print(f"[ok] Files in manifest: {len(manifest)}")

if __name__ == "__main__":
    main()
