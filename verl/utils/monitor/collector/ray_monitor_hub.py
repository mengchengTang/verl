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

"""Ray named actor that collects monitor events and exposes a metrics endpoint."""

from __future__ import annotations

import logging
from typing import Any

import ray

from ..constants import MonitorEventKind, MonitorTraceStatus
from ..server_utils.opentelemetry_utils import OpenTelemetryTraceCollector
from ..server_utils.prometheus_utils import MetricRegistry, start_metrics_http_server, update_prometheus_config
from verl.workers.config.rollout import PrometheusConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["MonitorHubActor"]


@ray.remote(max_concurrency=32)
class MonitorHubActor:
    """Central monitor hub for metric events and the shared scrape endpoint."""

    def __init__(
        self,
        port: int,
        addr: str | None = None,
        namespace: str = "",
        subsystem: str = "",
        histogram_buckets: tuple[float, ...] | None = None,
    ) -> None:
        self._registry = MetricRegistry(namespace=namespace, subsystem=subsystem)
        self._trace_collector = OpenTelemetryTraceCollector(namespace=namespace)
        self._histogram_buckets = histogram_buckets
        self._events_applied = 0
        self._event_handlers = {
            MonitorEventKind.COUNTER: self._handle_counter,
            MonitorEventKind.GAUGE: self._handle_gauge,
            MonitorEventKind.HISTOGRAM: self._handle_histogram,
            MonitorEventKind.TRACE: self._handle_trace,
        }

        bind_addr = (addr or "").strip()
        node_ip = ray.util.get_node_ip_address()
        scrape_host = bind_addr or node_ip
        start_metrics_http_server(port, addr=scrape_host)
        update_prometheus_config(PrometheusConfig(), [f"{scrape_host}:{port}"], name="trainer_metrics")

        listen_desc = scrape_host if scrape_host else "0.0.0.0"
        logger.info(
            "MonitorHubActor HTTP bind %s:%s, Prometheus scrape target %s:%s",
            listen_desc,
            port,
            scrape_host,
            port,
        )

    def apply_event(self, event: dict[str, Any]) -> None:
        """Apply a single monitor event to the in-process registry."""
        self._events_applied += 1
        try:
            kind = event["kind"]
        except KeyError as e:
            raise ValueError(f"Event missing required field: {e!r}") from e

        handler = self._event_handlers.get(kind)
        if handler is None:
            raise ValueError(f"Unknown event kind: {kind!r}")
        handler(event)

    @staticmethod
    def _require_fields(event: dict[str, Any], *fields: str) -> None:
        missing = [field for field in fields if field not in event]
        if missing:
            raise ValueError(f"Event missing required field(s): {', '.join(missing)}")

    def _handle_counter(self, event: dict[str, Any]) -> None:
        self._require_fields(event, "name", "value")
        self._registry.count(
            event["name"],
            event.get("documentation") or "",
            float(event["value"]),
            {},
            dict(event.get("labels") or {}),
        )

    def _handle_gauge(self, event: dict[str, Any]) -> None:
        self._require_fields(event, "name", "value")
        self._registry.value(
            event["name"],
            event.get("documentation") or "",
            float(event["value"]),
            {},
            dict(event.get("labels") or {}),
        )

    def _handle_histogram(self, event: dict[str, Any]) -> None:
        self._require_fields(event, "name", "value")
        self._registry.distribution(
            event["name"],
            event.get("documentation") or "",
            float(event["value"]),
            {},
            dict(event.get("labels") or {}),
            buckets=self._histogram_buckets,
        )

    def _handle_trace(self, event: dict[str, Any]) -> None:
        self._require_fields(event, "name", "span_key", "start_time_ns", "end_time_ns")
        self._trace_collector.record_span(
            event["span_key"],
            event["name"],
            int(event["start_time_ns"]),
            int(event["end_time_ns"]),
            status=str(event.get("status") or MonitorTraceStatus.OK),
            description=str(event.get("description") or ""),
            attributes=dict(event.get("attributes") or {}),
        )

    def get_stats(self) -> dict[str, Any]:
        """Return lightweight hub diagnostics."""
        return {
            "events_applied": self._events_applied,
            **self._trace_collector.get_stats(),
        }
