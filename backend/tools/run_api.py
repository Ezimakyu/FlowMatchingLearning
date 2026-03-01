from __future__ import annotations

import argparse
import os

from backend.app.config import load_env_file
from backend.app.logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FastAPI orchestration service.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable autoreload (development only).",
    )
    parser.add_argument(
        "--storage-dir",
        default=None,
        help="Optional override for job runtime storage directory.",
    )
    parser.add_argument(
        "--hyperparams-json",
        default=None,
        help="Optional override for hyperparameters JSON path.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional env file to load before execution (default: .env).",
    )
    parser.add_argument(
        "--no-env-file",
        action="store_true",
        help="Disable automatic env-file loading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    if not args.no_env_file:
        load_env_file(args.env_file, override=False)
    if args.storage_dir:
        os.environ["FLOW_JOB_STORAGE_DIR"] = args.storage_dir
    if args.hyperparams_json:
        os.environ["FLOW_HYPERPARAMS_JSON"] = args.hyperparams_json

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required to run the API. Install with `pip install uvicorn fastapi`."
        ) from exc

    uvicorn.run(
        "backend.app.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
