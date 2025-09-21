# JBI-Mental-Health — Review Bundle


This archive contains the *exact* code and artifacts referenced in the paper.
It was generated on **2025-09-20 20:58:32Z** by `make_review_bundle.py`.

## Layout

```
code/
  calibrate_model.py
  make_jbi_artifacts.py
  make_review_bundle.py
  optimize_primary_threshold.py
  retune_calibrated_safety_pconstr.py
  run_depr_jbi_eval_plus.py
  run_end_to_end.py
  train_tailboost_best.py
figures/
  Figure1_Calibration_Improvement.pdf
  Figure1_Calibration_Improvement.png
  Figure2_Threshold_Optimization.pdf
  Figure2_Threshold_Optimization.png
  Figure3_Framework_Overview.png
logs/
  20250920_200450_00_train_variants.log
  20250920_205432_10_threshold_primary_broad.log
  20250920_205433_11_threshold_primary_narrow.log
  20250920_205433_20_sweep.log
  20250920_205441_30_calibration_isotonic.log
  20250920_205445_40_pconstr_check.log
  20250920_205445_41_pconstr_COMMIT.log
paper_tables/
  Table1_Variant_Comparison.csv
  Table1_Variant_Comparison.tex
  Table3_Safety_Evaluation.csv
  Table3_Safety_Evaluation.tex
  Table4_Per_Label_Performance.csv
  Table4_Per_Label_Performance.tex
requirements.txt
results/
  safety_bh/
    calibration_evaluation.csv
    calibration_plots/
      anger_calibration.png
      brain_dysfunction_calibration.png
      emptiness_calibration.png
      hopelessness_calibration.png
      loneliness_calibration.png
      sadness_calibration.png
      suicide_intent_calibration.png
      worthlessness_calibration.png
    final_metrics_test.json
    label_order.json
    ops_safety_fdr.json
    per_label_report.csv
    primary_label_ci.json
    safety_calibrated.csv
    summary.json
    test_labels.npy
    test_logits.npy
    test_mask.npy
    test_probs_calibrated.npy
    thresholds.json
    tuned_thresholds.npy
    val_labels.npy
    val_logits.npy
    val_mask.npy
    val_probs_calibrated.npy
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
