# Hyperparameters Guide

The project keeps central tuning values in:

- `backend/config/hyperparameters.json`

This file uses regular JSON plus `_comment` keys (safe metadata) to explain each field.

## Current sections

- `phase_a`
  - `max_vision_chunk_tokens`: chunk size for VLM-extracted text.
  - `max_transcript_chunk_tokens`: chunk size for Whisper transcript text.
  - `include_transcript_chunks`: include transcript chunks in Phase A output.

- `phase_b.toc_generation`
  - `max_input_chars`: max text passed to TOC reasoning call.
  - `model_test`: default model for test runs.
  - `model_demo`: default model for demo runs.
  - `prompt_version`: versioned prompt key.
  - `temperature`: reasoning randomness.
  - `max_output_tokens`: max generation budget.

- `phase_b.iteration_loop`
  - Reserved for section parsing + edge-linking loop tuning.
  - Values are already defined for upcoming implementation.

## CLI usage

Both scripts accept a hyperparameter file path:

```bash
python backend/tools/run_phase_a.py --hyperparams-json backend/config/hyperparameters.json ...
python backend/tools/run_phase_b_toc.py --hyperparams-json backend/config/hyperparameters.json ...
```

For TOC model selection:

```bash
# Uses model_test
python backend/tools/run_phase_b_toc.py --model-profile test ...

# Uses model_demo
python backend/tools/run_phase_b_toc.py --model-profile demo ...

# Explicit override wins
python backend/tools/run_phase_b_toc.py --model gpt-4.1-mini ...
```
