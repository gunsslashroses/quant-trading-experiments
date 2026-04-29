# CTF NN Submission Guide

This folder contains a submission-ready script:
- `nn_ctf_submission.py`

It exposes the required CTF function:
- `main(chars, features, daily_ret) -> DataFrame[id, eom, w]`

## 1) Edit configuration

Open `nn_ctf_submission.py` and edit the `CONFIG = SubmissionConfig()` block.

Most important knobs:
- Universe: `use_us_only`, `require_common`
- Train window: `train_window_mode`, `rolling_train_years`, `min_train_months_required`
- Date guards: `min_train_eom`, `max_train_eom`, `min_test_eom`, `max_test_eom`
- Target preprocessing: `winsorize_target`, `winsor_lower_q`, `winsor_upper_q`
- Portfolio split: `lower_pct`, `upper_pct` (30/70, 50/50, etc.)
- Weighting: `weight_method = "equal"` or `"char_rank_weighted"`
- NN params: layers, activation, l2, lr, epochs, batch size

## 2) Generate the CSV locally

Run:

```bash
python ctf_submission/nn_ctf_submission.py \
  --chars /path/to/ctff_chars.parquet \
  --features /path/to/ctff_features.parquet \
  --daily-ret /path/to/ctff_daily_ret.parquet \
  --output-csv /path/to/weights_nn_us_30_70.csv
```

This writes the **exact CSV** you upload to CTF.

## 3) Files to upload on CTF submit page

Required:
1. **Model Script**: `ctf_submission/nn_ctf_submission.py`
2. **Model Weights CSV**: your generated CSV (e.g. `weights_nn_us_30_70.csv`)

Optional:
- Dependencies file (`requirements.txt` or `pyproject.toml`)
- Documentation PDF

## FAQ

### Do I need Excel?
No. CTF needs a **CSV** file. The script creates it directly.

### Can the code create the CSV for me?
Yes. Use `--output-csv ...` and upload that file.

### Can I switch 30/70 to 50/50?
Yes. Set:
- `lower_pct=0.50`
- `upper_pct=0.50`

(Though in practice you may prefer a small gap like 49/51 to avoid overlap ties.)
