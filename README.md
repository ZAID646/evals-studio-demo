---
title: Evals Studio Demo
emoji: 📊
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.36.2
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# Evals Studio Demo

A multi-provider LLM evaluation framework — generate completions, score them programmatically, and compare models on a live leaderboard.

**[→ Live Demo](https://zaid646-evals-studio-demo.hf.space)**

---

## Overview

Run standardized benchmarks across LLM providers, measuring accuracy, token-level F1, and ROUGE-L similarity. Uses **LLM-as-judge** for semantic evaluation.

### Benchmarks

| Benchmark | Description | Samples |
|---|---|---|
| `mmlu` | General knowledge (multiple-choice) | 8 |
| `factual_consistency` | Summarization factual accuracy | 3 |
| `instruction_following` | Exact instruction adherence | 3 |

### Metrics

- **Accuracy** — Exact string match (case-insensitive, punctuation-normalized)
- **F1 Score** — Token-level precision & recall between predicted and reference
- **ROUGE-L** — Longest Common Subsequence based recall

---

## Running Locally

```bash
git clone https://github.com/zaid646/evals-studio-demo.git
cd evals-studio-demo
pip install -r requirements.txt
python app.py
```

### Required Environment Variables

| Variable | Description |
|---|---|
| `NVIDIA_API_KEY` | NVIDIA API key (default provider) |

---

## Architecture

```
User selects model + benchmark
        ↓
  Generate completions via API
        ↓
  Score each against reference
  (accuracy / F1 / ROUGE-L)
        ↓
  LLM-as-judge evaluates quality
        ↓
  Leaderboard table (session state)
```

---

## License

MIT — see [LICENSE](LICENSE).
