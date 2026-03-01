# Actian VectorAI Setup (Hackathon)

## Where to update Compose

Update this file in your local clone:

- `actian-vectorAI-db-beta/docker-compose.yml`

This is the Docker Compose file that controls container image, ports, and persistence mounts.

## Current recommended compose service

```yaml
services:
  vectoraidb:
    image: williamimoh/actian-vectorai-db:1.0b
    container_name: vectoraidb
    platform: linux/amd64
    ports:
      - "50051:50051"
    volumes:
      - ./data:/data
    restart: unless-stopped
```

Notes:

- `./data:/data` persists collections and logs on your host.
- `platform: linux/amd64` is important on Apple Silicon when running this image.
- `docker logs vectoraidb` can be empty. The server may log to `/data/vde.log` when available.

## Python environment strategy

- Keep your app code in Conda env `modal`.
- Keep VectorAI DB in Docker.
- Connect from Python with `CortexClient("localhost:50051")`.

## Install the Actian Python client

The hackathon repo expects the `cortex` package from a wheel:

```bash
pip install ./actiancortex-0.1.0b1-py3-none-any.whl
```

If the wheel is not in the repo clone, ask the organizers for the wheel file and install from its path.

## Troubleshooting: gRPC "connection reset by peer" on Apple M4

If quick start fails with:

- `StatusCode.UNAVAILABLE`
- `recvmsg:Connection reset by peer`

and the container is running, this is likely the known Apple M4 + Rosetta issue.

### Fix

1. Open Docker Desktop settings.
2. Disable **Use Rosetta for x86/amd64 emulation on Apple Silicon**.
3. Restart Docker Desktop.
4. Recreate the container:

```bash
cd actian-vectorAI-db-beta
docker compose down
docker compose up -d --force-recreate
```

5. Re-test from Conda `modal` env:

```bash
python examples/quick_start.py localhost:50051
```

### Sanity check

After restart, this process should no longer be launched via `/run/rosetta/rosetta`:

```bash
docker exec vectoraidb ps aux
```

## Backend wiring in this project

Phase A now uses `backend/app/storage/actian_cortex_store.py` with these env vars:

- `ACTIAN_VECTORAI_ADDR` (default `localhost:50051`)
- `ACTIAN_COLLECTION_PREFIX` (default `course_chunks`)
- `ACTIAN_DISTANCE_METRIC` (default `COSINE`)
- `ACTIAN_HNSW_M` (default `16`)
- `ACTIAN_HNSW_EF_CONSTRUCT` (default `200`)
- `ACTIAN_HNSW_EF_SEARCH` (default `50`)
