from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 8.0


def run_with_retry(
    *,
    operation_name: str,
    fn: Callable[[], T],
    policy: RetryPolicy,
    logger: logging.Logger,
) -> T:
    if policy.max_attempts < 1:
        raise ValueError("RetryPolicy.max_attempts must be >= 1.")

    backoff = max(0.0, float(policy.initial_backoff_seconds))
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            is_last_attempt = attempt >= policy.max_attempts
            if is_last_attempt:
                logger.error(
                    "retry.failed operation=%s attempts=%d error=%s",
                    operation_name,
                    attempt,
                    exc.__class__.__name__,
                )
                raise
            logger.warning(
                "retry.retrying operation=%s attempt=%d max_attempts=%d sleep_s=%.2f error=%s",
                operation_name,
                attempt,
                policy.max_attempts,
                backoff,
                exc.__class__.__name__,
            )
            if backoff > 0:
                time.sleep(backoff)
            backoff = min(
                policy.max_backoff_seconds,
                max(backoff * policy.backoff_multiplier, policy.initial_backoff_seconds),
            )
