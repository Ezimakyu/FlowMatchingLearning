# FlowMatchingLearning


(Note to later that for project setup, will need to run this)
modal deploy backend/modal/transcription_service.py
modal deploy backend/modal/vision_extraction_service.py
modal deploy backend/modal/embedding_service.py

Understanding some params:
--doc-id -> document id (drives chunk ids)
--source-file-id -> id for file (to trace)
--model-profile (test, demo) 

Phase B end-to-end to graph:
python backend/tools/run_phase_b_toc.py \
  --phase-a-json artifacts/phase_a_result.json \
  --output-toc-json artifacts/toc.json \
  --output-meta-json artifacts/toc_meta.json \
  --model-profile test

python backend/tools/run_phase_b_graph.py \
  --phase-a-json artifacts/phase_a_result.json \
  --toc-json artifacts/toc.json \
  --output-graph-json artifacts/graph_data.json \
  --output-rolling-state-json artifacts/rolling_state.json \
  --output-section-results-json artifacts/section_parse_results.json \
  --model-profile test

Single-command full pipeline (Phase A -> TOC -> Graph):
python backend/tools/run_full_pipeline.py \
  --doc-id sample \
  --source-file sample_data/sample_lecture.pdf \
  --source-file-id sample_lecture \
  --output-dir artifacts/full_run \
  --model-profile test

Step 5 API (orchestration + state machine):
pip install -e ".[api]"
python backend/tools/run_api.py --host 127.0.0.1 --port 8000

Key endpoints:
- POST /api/v1/upload
- POST /api/v1/jobs/start
- GET /api/v1/jobs/{job_id}
- GET /api/v1/jobs/{job_id}/graph
- POST /api/v1/jobs/{job_id}/export

Step 6 frontend (React Flow):
conda activate modal
conda install -c conda-forge nodejs -y
cd frontend
npm install
npm run dev

Frontend notes:
- Default data source is frontend/public/graph_data.json
- API graph loading expects backend API running on http://127.0.0.1:8000
- Vite proxies /api/* to backend/tools/run_api.py server