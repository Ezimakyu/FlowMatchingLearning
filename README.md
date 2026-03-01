# FlowMatchingLearning

Creator: Stephen Zhu

## Quick setup and run

This project uses:
- **Phase A ingestion** on Modal + Actian VectorAI storage
- **Phase B reasoning** with OpenAI
- **FastAPI backend** + **React frontend**

Follow this once on a new machine.

## 1) Create and activate Conda environment

```bash
conda create -n modal python=3.11 -y
conda activate modal
python --version
```

If Node.js is not installed on your machine, install it in the same env:

```bash
conda install -c conda-forge nodejs -y
node --version
npm --version
```

## 2) Install Python + frontend packages

From repo root:

```bash
pip install --upgrade pip
pip install -e ".[phase_a,phase_b,api,dev]"
```

Install frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

### Actian client package (required for Phase A storage)

If you are running Phase A ingestion, install the Actian Cortex Python client wheel:

```bash
pip install /path/to/actiancortex-0.1.0b1-py3-none-any.whl
```

If you do not have the wheel file, ask the organizers and see `backend/docs/ACTIAN_SETUP.md`.

## 3) Modal setup and deployment

Authenticate Modal once:

```bash
modal setup
```

Deploy the three Phase A services (from repo root):

```bash
modal deploy backend/modal/transcription_service.py
modal deploy backend/modal/vision_extraction_service.py
modal deploy backend/modal/embedding_service.py
```

The backend expects these deployed Modal app/function names:
- `phase-a-transcription` / `transcribe_media`
- `phase-a-vision-extraction` / `extract_document_vision`
- `phase-a-embedding` / `embed_chunks`

## 4) API keys and `.env`

Create a `.env` file at repo root (auto-loaded by run scripts):

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Recommended defaults
OPENAI_REASONING_MODEL=gpt-4.1-mini
OPENAI_REASONING_TEMPERATURE=0.0
OPENAI_REASONING_MAX_OUTPUT_TOKENS=2500
TOC_PROMPT_VERSION=2026-02-28.v2
SECTION_CONCEPT_PROMPT_VERSION=2026-03-01.v3
EDGE_VALIDATION_PROMPT_VERSION=2026-03-01.v3

# Actian VectorAI (defaults shown)
ACTIAN_VECTORAI_ADDR=localhost:50051
ACTIAN_COLLECTION_PREFIX=course_chunks
ACTIAN_DISTANCE_METRIC=COSINE
ACTIAN_HNSW_M=16
ACTIAN_HNSW_EF_CONSTRUCT=200
ACTIAN_HNSW_EF_SEARCH=50
```

API keys/accounts you need:
- **OpenAI API key** (`OPENAI_API_KEY`) from OpenAI platform
- **Modal account auth** (set by running `modal setup`)

No extra API key is required for local Actian VectorAI by default.

## 5) Run everything (two terminal strategy)

Use the same `modal` conda env in both terminals.

### Terminal 1: start backend API

```bash
cd /Users/steph/Desktop/FlowMatchingLearning
conda activate modal
python backend/tools/run_api.py --host 127.0.0.1 --port 8000 --reload
```

### Terminal 2: start frontend

```bash
cd /Users/steph/Desktop/FlowMatchingLearning
conda activate modal
cd frontend
npm run dev
```

Then open:
- Frontend: `http://127.0.0.1:5173`
- Backend docs: `http://127.0.0.1:8000/docs`

Frontend requests to `/api/*` are proxied to the backend on port `8000`.

## 6) Helpful checks

Run preflight checks after setup:

```bash
python backend/tools/preflight.py --phase all --skip-actian
```

If your Actian service is already running and reachable:

```bash
python backend/tools/preflight.py --phase all
```

## 7) Useful API endpoints

- `POST /api/v1/upload`
- `POST /api/v1/jobs/start`
- `POST /api/v1/jobs/start-combined`
- `GET /api/v1/jobs/{job_id}`
- `GET /api/v1/jobs/{job_id}/graph`
- `POST /api/v1/jobs/{job_id}/export`

Hyperparameters file: `backend/config/hyperparameters.json`
