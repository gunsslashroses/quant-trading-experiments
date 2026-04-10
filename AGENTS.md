# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Python-based quantitative trading experiments project. Uses `uv` for fast dependency management with a virtual environment at `.venv/`.

### Quick reference

| Task | Command |
|------|---------|
| Install/update deps | `source .venv/bin/activate && uv pip install -e ".[dev]"` |
| Lint | `ruff check src/ tests/ examples/` |
| Format | `ruff format src/ tests/ examples/` |
| Test | `pytest tests/ -v` |
| Run demo | `python examples/sma_crossover_demo.py` |

### Caveats

- Always activate the venv first: `source .venv/bin/activate`
- `uv` is installed to `~/.local/bin`; ensure `PATH` includes it: `export PATH="$HOME/.local/bin:$PATH"`
- The project is installed in editable mode (`-e`), so source changes are picked up immediately without reinstall.
- No external services (databases, APIs) are required. Everything runs locally with synthetic data.
