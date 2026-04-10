# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Python-based quantitative trading research project. Factor models, signal combination, and ML return prediction on JKP data from WRDS. Uses `uv` for dependency management with a `.venv`.

### Quick reference

| Task | Command |
|------|---------|
| Install/update deps | `source .venv/bin/activate && uv pip install -e ".[dev]"` |
| Lint | `ruff check src/ tests/` |
| Format | `ruff format src/ tests/` |
| Test | `pytest tests/ -v` |

### Caveats

- Activate the venv first: `source .venv/bin/activate`
- `uv` is at `~/.local/bin`; ensure `PATH` includes it: `export PATH="$HOME/.local/bin:$PATH"`
- Installed in editable mode (`-e`); source changes take effect immediately.
- `pandas-datareader` requires `setuptools` on Python 3.12+ (distutils was removed). It's included in the install.
- `fetch_ff5_factors()` in `factors.py` does a lazy import of `pandas_datareader` to avoid import errors when that module has compatibility issues.
- Data files (`data/jkp_data.csv`) are not tracked in git. Notebooks won't run end-to-end without the JKP data from WRDS.
- ML notebook (04) requires optional `[ml]` extras: `tensorflow`, `torch`, `ipca`.
- No external services needed. Everything runs locally.
