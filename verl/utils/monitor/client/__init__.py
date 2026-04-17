from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["create_monitor_client"]


def create_monitor_client(
    *,
    port: int,
    addr: str | None = None,
    namespace: str = "",
    subsystem: str = "",
    histogram_buckets: tuple[float, ...] | None = None,
    wait_for_ack: bool = True,
) -> Any | None:
    """Create the configured monitor client."""
    try:
        from .ray_monitor_client import create_monitor_client as create_ray_monitor_client
    except ImportError as e:
        logger.warning("Ray monitor client is unavailable; monitoring is disabled: %s", e)
        return None

    return create_ray_monitor_client(
        port=port,
        addr=addr,
        namespace=namespace,
        subsystem=subsystem,
        histogram_buckets=histogram_buckets,
        wait_for_ack=wait_for_ack,
    )
