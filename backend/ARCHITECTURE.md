# Hybrid Microservices Architecture

This backend follows strict compute boundaries:

1. **Ingestion Layer (Modal serverless GPU functions)**
   - `ingestion.transcription`: Whisper-based audio/video transcription.
   - `ingestion.vision_extraction`: PDF/slide text + image description extraction via VLM.
   - `ingestion.embedding`: high-throughput chunk embedding generation.
   - Implemented in:
     - `backend/modal/transcription_service.py`
     - `backend/modal/vision_extraction_service.py`
     - `backend/modal/embedding_service.py`

2. **Storage Layer (Actian VectorDB)**
   - Stores extracted chunks and vectors.
   - Executes similarity search for historical concept retrieval.
   - Hackathon integration path uses the `cortex` client (`backend/app/storage/actian_cortex_store.py`).

3. **Reasoning Layer (OpenAI API only)**
   - `reasoning.toc_generation`
   - `reasoning.section_concept_extraction`
   - `reasoning.edge_validation`
   - Prompts are versioned in `backend/app/prompts/templates.py`.
   - Phase B TOC slice is implemented in:
     - `backend/app/reasoning/toc_reasoning.py`
     - `backend/app/pipelines/phase_b_toc.py`

4. **Presentation + Export Layer**
   - React Flow consumes `graph_data.json`.
   - Supermemory export persists finalized topology.

## Enforcement points

- `backend/app/models.py`
  - `LLMCallMetadata.provider` is fixed to `"openai"`.
  - `SimilarityMatch.retrieval_provider` is fixed to `"actian"`.
  - Ingestion outputs (`TranscriptionResult`, `VisionExtractionResult`, `EmbeddingBatchResult`) are fixed to provider `"modal"` and runtime `"serverless_gpu"`.
- `backend/app/compute_boundaries.py`
  - `assert_compute_boundary(stage, provider)` enforces stage-to-provider mapping.
- `backend/app/pipelines/phase_a.py`
  - orchestration enforces ingestion/storage provider checks before each stage.
- `backend/app/pipelines/phase_b_toc.py`
  - orchestration enforces reasoning provider checks for TOC generation.
- `backend/tools/preflight.py`
  - validates Python env, required modules, env keys, and Actian health check.

## Incremental tests

- Unit tests are centralized in `tests/unit/` and run with `pytest`.
- Current coverage includes:
  - chunking behavior
  - provider boundary checks
  - Phase A orchestration flow with fakes
  - Actian cortex adapter behavior
  - Phase B TOC input + pipeline behavior
  - TOC reasoning JSON extraction/client parsing behavior
