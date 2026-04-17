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

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any

from ..constants import MonitorTraceStatus

logger = logging.getLogger(__name__)

__all__ = [
    "OpenTelemetryTraceCollector",
    "OtelTraceCollector",
    "resolve_otlp_traces_endpoint",
]


def _require_opentelemetry() -> SimpleNamespace:
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.trace import Status, StatusCode
    except ImportError as e:
        raise ImportError(
            "OpenTelemetry trace export requires OpenTelemetry packages. "
            "Install with: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http "
            "or: pip install verl[monitoring]"
        ) from e

    return SimpleNamespace(
        OTLPSpanExporter=OTLPSpanExporter,
        Resource=Resource,
        SERVICE_NAME=SERVICE_NAME,
        Status=Status,
        StatusCode=StatusCode,
        TracerProvider=TracerProvider,
        SimpleSpanProcessor=SimpleSpanProcessor,
    )


def resolve_otlp_traces_endpoint() -> str:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if endpoint:
        return endpoint

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        return endpoint.rstrip("/") + "/v1/traces"

    return "http://127.0.0.1:4318/v1/traces"


def _normalize_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if not attributes:
        return normalized

    for key, value in attributes.items():
        key = str(key)
        if value is None:
            continue
        if isinstance(value, (bool, int, float, str)):
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


class OpenTelemetryTraceCollector:
    """Export complete root spans through OTLP/HTTP."""

    def __init__(self, namespace: str = "") -> None:
        self._spans_recorded = 0
        self._enabled = False
        self._otel = _require_opentelemetry()

        resource_attributes = {
            self._otel.SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", "verl-monitor"),
        }
        if namespace:
            resource_attributes["service.namespace"] = namespace

        provider = self._otel.TracerProvider(
            resource=self._otel.Resource.create(resource_attributes),
        )
        exporter = self._otel.OTLPSpanExporter(endpoint=resolve_otlp_traces_endpoint())
        provider.add_span_processor(self._otel.SimpleSpanProcessor(exporter))

        self._provider = provider
        self._tracer = provider.get_tracer(__name__)
        self._enabled = True

    def record_span(
        self,
        span_key: str,
        name: str,
        start_time_ns: int,
        end_time_ns: int,
        *,
        status: str = MonitorTraceStatus.OK,
        description: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if not self._enabled:
            return

        del span_key
        span = self._tracer.start_span(
            name=name,
            start_time=start_time_ns,
            attributes=_normalize_attributes(attributes),
        )

        if status == MonitorTraceStatus.ERROR:
            span.set_status(self._otel.Status(self._otel.StatusCode.ERROR, description))

        span.end(end_time=end_time_ns)
        self._spans_recorded += 1

    def get_stats(self) -> dict[str, int]:
        return {
            "trace_spans_recorded": self._spans_recorded,
        }


OtelTraceCollector = OpenTelemetryTraceCollector
