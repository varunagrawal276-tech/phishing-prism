# PRISM-Phish

**Source-Invariant and Perturbation-Consistent Phishing Email Detection Using Hybrid Transformer Representations**

A research-grade machine learning system for robust cross-dataset phishing email detection.

## Research Question

> Can a lightweight domain-specialized phishing detector learn phishing-relevant representations
> that generalize across heterogeneous email datasets better than conventional classifiers,
> standard Transformer classifiers, and general-purpose LLM baselines?

## Architecture

PRISM-Phish combines:
- **DeBERTa-v3-small** contextual text encoder
- **Deterministic security/structural features** (URL, lexical, header, format)
- **Gated fusion** mechanism
- **Domain-adversarial training** via gradient reversal (source-invariant representations)
- **Perturbation consistency** training (stability under label-preserving transformations)
- **Calibrated output** with selective prediction / abstention support

## Dataset

Seven curated phishing email datasets from [Figshare](https://figshare.com/articles/dataset/Seven_Phishing_Email_Datasets/25432108):
- Ling, Enron, Assassin, TREC-05, TREC-06, TREC-07, CEAS-08
- ~203K curated email instances

## Quick Start

```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -e ".[dev]"

# Run dataset audit
python scripts/audit_dataset.py

# Train baselines
python scripts/train_baselines.py --config configs/training.yaml

# Train PRISM-Phish
python scripts/train_prism.py --config configs/training.yaml

# Run evaluation
python scripts/run_cross_source.py --config configs/evaluation.yaml
```

## Project Structure

```
phishing-prism/
├── configs/          # YAML configuration files
├── data/             # Raw, interim, processed data and splits
├── artifacts/        # Generated reports, checkpoints, plots
├── experiments/      # Experiment registry and results
├── notebooks/        # Analysis notebooks (call src/ modules)
├── src/              # Core source code
│   ├── data/         # Loading, schemas, deduplication, splitting
│   ├── preprocessing/# Text, HTML, URL preprocessing
│   ├── features/     # Lexical, structural, URL, header features
│   ├── models/       # Classical, transformer, fusion, PRISM-Phish
│   ├── training/     # Trainer, losses, callbacks, checkpointing
│   ├── evaluation/   # Metrics, calibration, cross-source, reports
│   ├── robustness/   # Perturbations, prompt injection, attack metrics
│   ├── explainability/ # SHAP, token attribution, error analysis
│   ├── llm/          # LLM benchmark adapters and prompts
│   └── api/          # FastAPI inference endpoint
├── scripts/          # CLI entry points
├── tests/            # Unit and integration tests
└── app/              # Frontend demo
```

## Safety & Ethics

This is a **defensive cybersecurity research project**. It does not send phishing emails,
target organizations, collect credentials, or deploy malicious payloads. All email contents
are treated as untrusted text.

## Citation

If using the dataset, cite:
```bibtex
@inproceedings{champa2024curated,
  title={Curated Datasets and Feature Analysis for Phishing Email Detection with Machine Learning},
  author={Champa, Arifa I and Rabbi, Md Fazle and Zibran, Minhaz F},
  booktitle={3rd IEEE International Conference on Computing and Machine Intelligence (ICMI)},
  pages={1--7},
  year={2024}
}
```

## License

MIT
