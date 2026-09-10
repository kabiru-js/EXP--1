# Changelog

## [0.4.0] - 2026-09-10

### Research-grade restructure
- Added `AUDIT.md` — comprehensive repository audit
- Added `research/` directory with `QUESTION.md`, `METHODOLOGY.md`, `RELATED_WORK.md`, `ROADMAP.md`, `REPRODUCIBILITY.md`, `LIMITATIONS.md`
- Added `experiments/EXP-001/README.md` and `experiments/EXP-002/README.md`
- Added `scripts/make_figures.py` — standalone figure generation from results JSONs
- Added `figures/` root directory for publication-quality figures
- Rewrote `README.md` as a research project page
- Updated `RESULTS.md` with complete results, CV table, honest interpretation
- Added `scripts/scratch/` probe scripts documentation

### Previous

## [0.3.0] - 2026-09-09

### Added
- EXP-002 onset diagnostic: subsequence-based onset detection on token-level signals
- `scripts/exp002_qwen.py` (Qwen CPU experiment script, deferred)
- `experiments/EXP-002/exp002_results.json` and `reports/EXP-002.md`

### Findings
- EXP-001: GBT AUROC=1.000 single split; 5-fold CV confirms AUROC=0.9946±0.0061 (genuine signal)
- EXP-001 limitations documented: correctness commits at token 0-1, no distinct pre-error degradation phase
- EXP-002: distilgpt2 produces onset at term 0 in 100% of 45 arithmetic-chain probes; mid-chain onset count = 0. Qwen2.5-0.5B-Instruct not benchmarked (not cached, no NVIDIA driver, CPU-only PyTorch)

## [0.2.0] - 2026-09-09

### Added
- `synthetic_sequences` dataset (sequence-continuation benchmark)
- `prompt_template` support in experiment configs and inference pipeline
- CPU threading configuration in the HuggingFace backend
- Trajectory storage `clear()` and pipeline resume support
- Cross-validation sanity-check script (`scripts/cv_check_exp001.py`)

### Completed
- EXP-001 full 1000-sample run: 466 correct / 534 incorrect (46.6%)
- Baselines: entropy_threshold AUROC=0.868, logistic_regression AUROC=0.975,
  gradient_boosted_tree AUROC=1.000 (single split); 5-fold CV confirms GBT
  AUROC=0.9946 +/- 0.0061
- Report, figures, and results JSON for EXP-001
- Honest limitations documented: correctness commits at token 0-1, so no
  distinct pre-error degradation phase in this benchmark

## [0.1.0] - 2026-09-09

### Added
- Project scaffolding and directory structure
- Configuration system (YAML-based experiment configs)
- Dataset loaders: GSM8K, TruthfulQA, synthetic arithmetic
- Model abstraction with HuggingFace backend
- Token-by-token generation instrumentation
- Parquet + DuckDB trajectory storage
- Feature extraction (aggregate and temporal)
- Baseline models: entropy threshold, logistic regression, gradient-boosted tree
- Divergence analysis and temporal prediction
- Publication-quality visualization
- Experiment reporting system
- CLI interface
- Test suite
- EXP-001 configuration
