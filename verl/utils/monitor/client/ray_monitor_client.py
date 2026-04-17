#!/usr/bin/env python
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ray client for the shared monitor hub."""

from __future__ import annotations

import logging
from typing import Any

import ray

from ..collector.ray_monitor_hub import MonitorHubActor

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["MonitorRayClient", "create_monitor_client"]

DEFAULT_MONITOR_HUB_ACTOR_NAME = "MonitorHubActor"


def _get_or_create_monitor_hub(
    port: int,
    addr: str | None = None,
    namespace: str = "",
    subsystem: str = "",
    histogram_buckets: tuple[float, ...] | None = None,
    actor_name: str = DEFAULT_MONITOR_HUB_ACTOR_NAME,
) -> Any:
    """Return a handle to the named monitor hub actor, creating it if missing."""
    if not ray.is_initialized():
        raise RuntimeError("Ray is not initialized. Call ray.init() before using monitor helpers.")

    logger.info("Getting or creating monitor hub actor %r.", actor_name)
    return MonitorHubActor.options(name=actor_name, get_if_exists=True).remote(
        port, addr, namespace, subsystem, histogram_buckets
    )


def create_monitor_client(
    *,
    port: int,
    addr: str | None = None,
    namespace: str = "",
    subsystem: str = "",
    histogram_buckets: tuple[float, ...] | None = None,
    wait_for_ack: bool = True,
) -> "MonitorRayClient | None":
    """Create the default Ray-backed monitor client when Ray is available."""
    if not ray.is_initialized():
        logger.warning("Ray is not initialized; monitoring is disabled.")
        return None

    handle = _get_or_create_monitor_hub(
        port=port,
        addr=addr,
        namespace=namespace,
        subsystem=subsystem,
        histogram_buckets=histogram_buckets,
    )
    return MonitorRayClient(handle, wait_for_ack=wait_for_ack)


class MonitorRayClient:
    """Thin wrapper that sends monitor events to ``MonitorHubActor``."""

    def __init__(self, actor_handle: Any, *, wait_for_ack: bool = True) -> None:
        self._actor = actor_handle
        self._wait_for_ack = wait_for_ack

    def apply_event(self, event: dict[str, Any]) -> None:
        """Send one event to the monitor hub."""
        ref = self._actor.apply_event.remote(event)
        if self._wait_for_ack:
            ray.get(ref)

    def get_stats(self) -> dict[str, Any]:
        """Fetch hub diagnostics via RPC."""
        return ray.get(self._actor.get_stats.remote())
