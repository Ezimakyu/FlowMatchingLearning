from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    details: str


def _result(name: str, ok: bool, details: str) -> CheckResult:
    return CheckResult(name=name, ok=ok, details=details)


def check_python_version() -> CheckResult:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 11)
    return _result("python_version", ok, f"{major}.{minor} (requires >= 3.11)")


def check_module_import(module_name: str) -> CheckResult:
    try:
        __import__(module_name)
        return _result(f"import_{module_name}", True, "installed")
    except Exception as exc:
        return _result(f"import_{module_name}", False, f"missing or broken: {exc}")


def check_env_key(key: str) -> CheckResult:
    value = os.getenv(key, "").strip()
    if value:
        return _result(f"env_{key}", True, "set")
    return _result(f"env_{key}", False, "not set")


def check_actian_health(address: str) -> CheckResult:
    try:
        from cortex import CortexClient
    except Exception as exc:
        return _result("actian_health", False, f"cortex client unavailable: {exc}")

    try:
        with CortexClient(address) as client:
            version, uptime = client.health_check()
        return _result("actian_health", True, f"connected: {version}, uptime={uptime}")
    except Exception as exc:
        return _result("actian_health", False, f"connection failed: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project preflight checks.")
    parser.add_argument(
        "--phase",
        choices=["phase_a", "phase_b", "all"],
        default="all",
        help="Select which checks to run.",
    )
    parser.add_argument(
        "--actian-addr",
        default=os.getenv("ACTIAN_VECTORAI_ADDR", "localhost:50051"),
        help="Actian VectorAI address (default: ACTIAN_VECTORAI_ADDR or localhost:50051).",
    )
    parser.add_argument(
        "--skip-actian",
        action="store_true",
        help="Skip Actian health check.",
    )
    return parser.parse_args()


def collect_checks(args: argparse.Namespace) -> list[CheckResult]:
    checks: list[CheckResult] = [check_python_version()]

    if args.phase in ("phase_a", "all"):
        checks.append(check_module_import("modal"))
        checks.append(check_module_import("cortex"))
        if not args.skip_actian:
            checks.append(check_actian_health(args.actian_addr))

    if args.phase in ("phase_b", "all"):
        checks.append(check_module_import("openai"))
        checks.append(check_env_key("OPENAI_API_KEY"))

    return checks


def print_results(results: list[CheckResult]) -> None:
    print("Preflight results:")
    for item in results:
        status = "OK" if item.ok else "FAIL"
        print(f"- [{status}] {item.name}: {item.details}")


def main() -> None:
    args = parse_args()
    results = collect_checks(args)
    print_results(results)
    failed = [item for item in results if not item.ok]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
