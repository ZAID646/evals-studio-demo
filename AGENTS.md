# evals-studio-demo

## Session Context (Jul 2026)

### Status
- Deployed at: https://zaid646-evals-studio-demo.hf.space
- Working: completion generation, LLM-as-judge metrics, leaderboard

### Key Decisions
- Uses NVIDIA minimax-m3 for both generation and evaluation
- Web UI via Gradio with JS-embedded evaluation viewer

### Secrets Required (HF Space)
- `NVIDIA_API_KEY`

### Fixed Bugs
1. Off-by-one stripping in metrics — `normalize()` was stripping trailing punctuation after stripping whitespace, causing empty strings for already-clean text. Fixed by checking for length > 0 before stripping.

### Architecture
- `app.py`: Gradio UI + OpenAI client calls
- Judges use NVIDIA minimax-m3 with structured prompts
- Results cached in session state (no persistence)
