# Backend Data Contracts

These JSON files define the strict contracts for pipeline artifacts:

- `graph_data.json`
- `rolling_state.json`
- `toc.json`
- `section_parse_result.json`
- `transcription_result.json`
- `vision_extraction_result.json`
- `embedding_batch_result.json`
- `chunking_result.json`
- `phase_a_ingestion_result.json`

They are JSON Schema documents (Draft 2020-12) that mirror `backend/app/models.py`.

Notes:

- Edge direction convention is fixed as `source=prerequisite`, `target=dependent`.
- DAG cycle rejection is enforced in Python validation (`GraphData` / `RollingState`).
- Compute boundaries are strict:
  - ingestion (`transcription`, `vision_extraction`, `embedding`) -> `modal`
  - storage / similarity retrieval -> `actian`
  - inner parsing loop reasoning (`toc_generation`, `section_concept_extraction`, `edge_validation`) -> `openai`
  - export -> `supermemory`
- To regenerate schemas from Pydantic models, run:
  - `python backend/tools/export_contract_schemas.py`
