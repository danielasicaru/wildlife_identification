"""In-process request metrics -- request count, error count, and a bounded window of recent
latencies for basic average/p50/p95 reporting. Not wired to Prometheus/Grafana; sufficient for a
single-process local deployment at this project's scale."""
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RequestMetrics:
    request_count: int = 0
    error_count: int = 0
    # Bounded so a long-running process doesn't grow this unboundedly; recent latencies are what
    # matters for "is it slow right now," not the full lifetime history.
    recent_latencies_ms: deque = field(default_factory=lambda: deque(maxlen=500))

    def record(self, status_code: int, duration_ms: float) -> None:
        self.request_count += 1
        if status_code >= 400:
            self.error_count += 1
        self.recent_latencies_ms.append(duration_ms)

    def summary(self) -> dict:
        if not self.recent_latencies_ms:
            return {
                "request_count": self.request_count, "error_count": self.error_count,
                "avg_latency_ms": None, "p50_latency_ms": None, "p95_latency_ms": None,
            }
        latencies = sorted(self.recent_latencies_ms)
        return {
            "request_count": self.request_count, "error_count": self.error_count,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "p50_latency_ms": round(_percentile(latencies, 0.50), 1),
            "p95_latency_ms": round(_percentile(latencies, 0.95), 1),
        }


def _percentile(sorted_values: list[float], fraction: float) -> float:
    index = min(int(len(sorted_values) * fraction), len(sorted_values) - 1)
    return sorted_values[index]
