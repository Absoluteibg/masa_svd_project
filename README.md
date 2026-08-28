# MASA-SVD: Mitigating LLM Hallucinations via Multi-Agent Subspace Alignment and SVD

**Anna University Regulation 2021 — Final Year Project (CSE)**

## Quick Start

```bash
# 1. Clone / copy this folder to your laptop
cd masa_svd_project

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Set up your API key
cp .env.example .env
# Edit .env and paste your OpenAI API key

# 5. Verify setup
python scripts/setup_check.py

# 6. Run demo
python scripts/run_demo.py --fast    # fast mode (no NLI download)
python scripts/run_demo.py           # full mode

# 7. Run tests
python tests/run_all_tests.py

# 8. Run full evaluation (generates your report tables)
python scripts/run_evaluation.py --n 50

# 9. Run ablation study (generates Table 5.4 in your report)
python scripts/run_ablation.py --n 20

# 10. Start API server (for viva demo)
python scripts/run_api.py
# Then open http://localhost:8000/docs
```

## Project Structure

See codebase document for full file tree and explanations.

## Results Location

All outputs saved to `results/`:
- `truthfulqa_results.csv` — per-question results
- `ablation_k.csv`         — k-value ablation
- `ablation_weights.csv`   — weight ablation
- `figures/`               — PNG plots for report

## Cost Estimate

- 50 TruthfulQA samples × 2 calls × ~$0.0002 = ~$0.02
- 20 ablation samples   × 6 configs × 2 calls = ~$0.05
- Total for all experiments: **< $0.15 USD**