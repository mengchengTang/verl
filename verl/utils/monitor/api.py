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

"""High-level monitor API backed by a pluggable monitor client."""

from __future__ import annotations

import functools
import logging
import os
import secrets
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generator

from .client import create_monitor_client
from .constants import MonitorEventKind, MonitorTraceStatus
from .server_utils.prometheus_utils import merge_labels

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = [
    "close",
    "init",
    "log",
    "metric_count",
    "metric_distribution",
    "metric_value",
    "trace_op",
    "trace_state",
]


@dataclass
class _MonitorState:
    """Internal module state for ``init`` / event helpers."""

    enabled: bool = False
    client: Any | None = None
    namespace: str = ""


_STATE = _MonitorState()


def init(namespace: str = "") -> None:
    """Initialize monitoring with the configured monitor client.

    The public initialization surface is intentionally minimal; advanced hub settings use
    internal defaults in the concrete client implementation.

    Args:
        namespace: Metric namespace on the hub.
    """
    global _STATE
    if _STATE.enabled:
        warnings.warn(
            "monitor.init() called more than once; ignoring re-initialization.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    port = int(os.environ.get("PROMETHEUS_METRICS_PORT", "9092"))
    client = create_monitor_client(port=port, namespace=namespace)
    _STATE = _MonitorState(
        enabled=client is not None,
        client=client,
        namespace=namespace,
    )


def close() -> None:
    """Reset module state.

    Does not stop the HTTP endpoint thread or kill the Ray hub actor.
    """
    global _STATE
    _STATE = _MonitorState()


def _emit(
    kind: str,
    name: str,
    value: float,
    documentation: str,
    labels: dict[str, Any],
) -> None:
    if not _STATE.enabled or _STATE.client is None:
        return
    merged = merge_labels(None, labels)
    doc = documentation or ""
    event = {
        "kind": kind,
        "name": name,
        "documentation": doc,
        "value": value,
        "labels": merged,
    }
    _STATE.client.apply_event(event)


def _trace_attributes(user: dict[str, Any] | None = None) -> dict[str, Any]:
    """Default span attributes for monitor traces.

    ``process_id`` defaults to the current OS PID (string). Caller-supplied keys in
    ``user`` override defaults on collision (e.g. explicit ``process_id``).
    """
    base: dict[str, Any] = {"process_id": str(os.getpid())}
    if not user:
        return base
    return {**base, **user}


def _emit_trace_span(
    *,
    span_key: str,
    name: str,
    start_time_ns: int,
    end_time_ns: int,
    attributes: dict[str, Any],
    status: str = MonitorTraceStatus.OK,
    description: str = "",
) -> None:
    if not _STATE.enabled or _STATE.client is None:
        return

    event = {
        "kind": MonitorEventKind.TRACE,
        "name": name,
        "span_key": span_key,
        "start_time_ns": int(start_time_ns),
        "end_time_ns": int(end_time_ns),
        "attributes": merge_labels(None, _trace_attributes(attributes)),
        "status": status,
        "description": description,
    }
    _STATE.client.apply_event(event)


def _new_span_key() -> str:
    return secrets.token_hex(8)


def metric_count(name: str, amount: float = 1.0, documentation: str = "", **labels: Any) -> None:
    """Record a counter increment.

    Args:
        name: Metric name.
        amount: Increment amount (typically 1.0).
        documentation: Help string; default derived from ``name``.
        **labels: Extra label key-values attached to the event.
    """
    doc = documentation or f"Counter {name}"
    _emit(MonitorEventKind.COUNTER, name, float(amount), doc, labels)


def metric_value(name: str, value: float, documentation: str = "", **labels: Any) -> None:
    """Record the latest value for a metric.

    Args:
        name: Metric name.
        value: Current value.
        documentation: Help string.
        **labels: Extra labels attached to the event.
    """
    doc = documentation or f"Gauge {name}"
    _emit(MonitorEventKind.GAUGE, name, float(value), doc, labels)


def metric_distribution(name: str, value: float, documentation: str = "", **labels: Any) -> None:
    """Record one sample into a metric distribution.

    Args:
        name: Metric name.
        value: Observed sample.
        documentation: Help string.
        **labels: Extra labels attached to the event.
    """
    doc = documentation or f"Histogram {name}"
    _emit(MonitorEventKind.HISTOGRAM, name, float(value), doc, labels)


def trace_op(
    name: str | None = None,
    *,
    extra_labels: Callable[[Any], dict[str, Any]] | None = None,
    **static_labels: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for synchronous call tracing through the shared monitor hub.

    Attributes include ``process_id`` (see :func:`_trace_attributes`) unless overridden.

    Args:
        name: Span name; defaults to ``func.__qualname__``.
        extra_labels: Optional ``callable(first_positional_arg) -> dict`` merged into
            span labels after ``static_labels``. For bound methods, the first argument
            is typically ``self``.
        **static_labels: Extra stringifiable labels attached to the span.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            span_name = name or func.__qualname__
            labels: dict[str, Any] = dict(static_labels)
            if extra_labels is not None and args:
                labels.update(extra_labels(args[0]))

            span_key = _new_span_key()
            start_time_ns = time.time_ns()
            attributes = {"trace_kind": "duration", **labels}
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                _emit_trace_span(
                    span_key=span_key,
                    name=span_name,
                    start_time_ns=start_time_ns,
                    end_time_ns=time.time_ns(),
                    attributes={
                        **attributes,
                        "error.type": type(e).__name__,
                        "error.message": str(e),
                    },
                    status=MonitorTraceStatus.ERROR,
                    description=str(e),
                )
                raise
            else:
                _emit_trace_span(
                    span_key=span_key,
                    name=span_name,
                    start_time_ns=start_time_ns,
                    end_time_ns=time.time_ns(),
                    attributes=attributes,
                )
                return result

        return wrapper

    return decorator


def log(message: str, **fields: Any) -> None:
    """Structured log hook placeholder (no-op)."""
    del message, fields


@contextmanager
def trace_state(
    func_name: str,
    *,
    metric_name: str = "rank_state_active",
    state_label: str = "state",
    empty_state: str = "",
    active_value: float = 1.0,
    inactive_value: float = 0.0,
    documentation: str = "Rank active state (for Grafana state timeline)",
    **labels: Any,
) -> Generator[None, None, None]:
    """Record a named runtime state as a root span for Grafana timeline views."""

    del active_value, empty_state, inactive_value

    span_key = _new_span_key()
    start_time_ns = time.time_ns()
    attributes = {
        "trace_kind": "state",
        "state_name": func_name,
        "state_label": state_label,
        "metric_name": metric_name,
        "documentation": documentation,
        **labels,
    }

    try:
        yield
    except Exception as e:
        _emit_trace_span(
            span_key=span_key,
            name=func_name,
            start_time_ns=start_time_ns,
            end_time_ns=time.time_ns(),
            attributes={
                **attributes,
                "error.type": type(e).__name__,
                "error.message": str(e),
            },
            status=MonitorTraceStatus.ERROR,
            description=str(e),
        )
        raise
    else:
        _emit_trace_span(
            span_key=span_key,
            name=func_name,
            start_time_ns=start_time_ns,
            end_time_ns=time.time_ns(),
            attributes=attributes,
        )
