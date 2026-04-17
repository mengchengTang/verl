# Copyright 2026 Meituan Ltd. and/or its affiliates
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
from typing import Any, Mapping

import yaml

from verl.workers.config.rollout import PrometheusConfig

logger = logging.getLogger(__file__)
logger.setLevel(logging.WARNING)

__all__ = [
    "MetricRegistry",
    "merge_labels",
    "normalize_label_values",
    "start_metrics_http_server",
    "update_prometheus_config",
]


def _require_prometheus():
    try:
        from prometheus_client import Counter, Gauge, Histogram  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Training metrics require the `prometheus_client` package. "
            "Install with: pip install prometheus_client "
            "or: pip install verl[monitoring]"
        ) from e
    return __import__("prometheus_client", fromlist=["Counter", "Gauge", "Histogram"])


def normalize_label_values(labels: Mapping[str, Any]) -> dict[str, str]:
    """Prometheus label values must be strings."""
    return {str(k): str(v) for k, v in labels.items()}


def merge_labels(defaults: Mapping[str, Any] | None, overrides: Mapping[str, Any] | None) -> dict[str, str]:
    """Merge label dicts; ``overrides`` wins on key collision. Values are stringified."""
    out: dict[str, str] = {}
    if defaults:
        out.update(normalize_label_values(defaults))
    if overrides:
        out.update(normalize_label_values(overrides))
    return out


def start_metrics_http_server(port: int, addr: str = "") -> None:
    """Start Prometheus ``/metrics`` HTTP endpoint (served by ``prometheus_client``)."""
    try:
        from prometheus_client import start_http_server
    except ImportError as e:
        raise ImportError(
            "start_metrics_http_server requires `prometheus_client`. Install with: pip install prometheus_client"
        ) from e
    start_http_server(port, addr=addr)


class MetricRegistry:
    """Create and cache metrics keyed by (name, kind, sorted label names)."""

    def __init__(self, namespace: str = "", subsystem: str = "") -> None:
        self._namespace = namespace
        self._subsystem = subsystem
        self._counters: dict[tuple[str, tuple[str, ...]], object] = {}
        self._gauges: dict[tuple[str, tuple[str, ...]], object] = {}
        self._histograms: dict[tuple[str, tuple[str, ...]], object] = {}

    def _get_or_create_counter(self, name: str, documentation: str, label_names: tuple[str, ...]):
        prom = _require_prometheus()
        Counter = prom.Counter
        key = (name, label_names)
        if key not in self._counters:
            self._counters[key] = Counter(
                name,
                documentation,
                labelnames=label_names,
                namespace=self._namespace,
                subsystem=self._subsystem,
            )
        return self._counters[key]

    def _get_or_create_gauge(self, name: str, documentation: str, label_names: tuple[str, ...]):
        prom = _require_prometheus()
        Gauge = prom.Gauge
        key = (name, label_names)
        if key not in self._gauges:
            self._gauges[key] = Gauge(
                name,
                documentation,
                labelnames=label_names,
                namespace=self._namespace,
                subsystem=self._subsystem,
            )
        return self._gauges[key]

    def _get_or_create_histogram(
        self,
        name: str,
        documentation: str,
        label_names: tuple[str, ...],
        buckets: tuple[float, ...] | None,
    ):
        prom = _require_prometheus()
        Histogram = prom.Histogram
        key = (name, label_names)
        if key not in self._histograms:
            kw = {}
            if buckets is not None:
                kw["buckets"] = buckets
            self._histograms[key] = Histogram(
                name,
                documentation,
                labelnames=label_names,
                namespace=self._namespace,
                subsystem=self._subsystem,
                **kw,
            )
        return self._histograms[key]

    def count(
        self,
        name: str,
        documentation: str,
        amount: float,
        defaults: Mapping[str, Any] | None = None,
        labels: Mapping[str, Any] | None = None,
    ) -> None:
        merged = merge_labels(defaults, labels)
        names = tuple(sorted(merged.keys()))
        counter = self._get_or_create_counter(name, documentation, names)
        counter.labels(**merged).inc(amount)

    def value(
        self,
        name: str,
        documentation: str,
        value: float,
        defaults: Mapping[str, Any] | None = None,
        labels: Mapping[str, Any] | None = None,
    ) -> None:
        merged = merge_labels(defaults, labels)
        names = tuple(sorted(merged.keys()))
        gauge = self._get_or_create_gauge(name, documentation, names)
        if merged:
            gauge.labels(**merged).set(value)
        else:
            gauge.set(value)

    def distribution(
        self,
        name: str,
        documentation: str,
        value: float,
        defaults: Mapping[str, Any] | None = None,
        labels: Mapping[str, Any] | None = None,
        buckets: tuple[float, ...] | None = None,
    ) -> None:
        merged = merge_labels(defaults, labels)
        names = tuple(sorted(merged.keys()))
        histogram = self._get_or_create_histogram(name, documentation, names, buckets)
        histogram.labels(**merged).observe(value)


def update_prometheus_config(
    config: PrometheusConfig, server_addresses: list[str], name: str | None = None
) -> None:
    """
    Read ``config.file`` (YAML), set or replace the scrape job for ``job_name`` with
    ``static_configs`` targets ``server_addresses``, write to all Ray nodes, then reload Prometheus.
    ``job_name`` is ``name`` if given, otherwise ``"rollout"``.
    """

    if not server_addresses:
        logger.warning("No server addresses available to update Prometheus config")
        return

    job_name = name or "rollout"

    try:
        ray = __import__("ray")

        with open(config.file, encoding="utf-8") as f:
            prometheus_data = yaml.safe_load(f) or {}
        scrape_configs = prometheus_data.setdefault("scrape_configs", [])
        new_job = {"job_name": job_name, "static_configs": [{"targets": server_addresses}]}
        for i, sc in enumerate(scrape_configs):
            if sc.get("job_name") == job_name:
                scrape_configs[i] = new_job
                break
        else:
            scrape_configs.append(new_job)

        # Write configuration file to all nodes
        @ray.remote(num_cpus=0)
        def write_config_file(config_data, config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            return True

        # Reload Prometheus on all nodes. Only master node should succeed, skip errors on other nodes.
        @ray.remote(num_cpus=0)
        def reload_prometheus(port):
            import socket
            import subprocess

            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)

            reload_url = f"http://{ip_address}:{port}/-/reload"

            try:
                subprocess.run(["curl", "-X", "POST", reload_url], capture_output=True, text=True, timeout=10)
                print(f"Reloading Prometheus on node: {reload_url}")
            except Exception:
                # Skip errors on non-master nodes
                pass

        # Get all available nodes and schedule tasks on each node
        nodes = ray.nodes()
        alive_nodes = [node for node in nodes if node["Alive"]]

        # Write config files on all nodes
        write_tasks = []
        for node in alive_nodes:
            node_ip = node["NodeManagerAddress"]
            task = write_config_file.options(
                resources={"node:" + node_ip: 0.001}  # Schedule to specific node
            ).remote(prometheus_data, config.file)
            write_tasks.append(task)

        ray.get(write_tasks)

        print(
            f"Updated Prometheus configuration at {config.file} with {len(server_addresses)} targets (job_name={job_name})"
        )

        # Reload Prometheus on all nodes
        reload_tasks = []
        for node in alive_nodes:
            node_ip = node["NodeManagerAddress"]
            task = reload_prometheus.options(
                resources={"node:" + node_ip: 0.001}  # Schedule to specific node
            ).remote(config.port)
            reload_tasks.append(task)

        ray.get(reload_tasks)

    except Exception as e:
        logger.error(f"Failed to update Prometheus configuration: {e}")
