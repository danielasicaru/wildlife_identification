from src.api.metrics import RequestMetrics


def test_record_counts_requests_and_errors():
    metrics = RequestMetrics()

    metrics.record(200, 10.0)
    metrics.record(404, 5.0)
    metrics.record(500, 20.0)

    assert metrics.request_count == 3
    assert metrics.error_count == 2


def test_summary_reports_none_latencies_when_no_requests_recorded():
    metrics = RequestMetrics()

    summary = metrics.summary()

    assert summary["request_count"] == 0
    assert summary["avg_latency_ms"] is None


def test_summary_computes_avg_and_percentiles():
    metrics = RequestMetrics()
    for latency in [10.0, 20.0, 30.0, 40.0, 50.0]:
        metrics.record(200, latency)

    summary = metrics.summary()

    assert summary["avg_latency_ms"] == 30.0
    assert summary["p50_latency_ms"] == 30.0
    assert summary["p95_latency_ms"] == 50.0


def test_recent_latencies_are_bounded():
    metrics = RequestMetrics()
    for i in range(600):
        metrics.record(200, float(i))

    assert metrics.request_count == 600  # total count is not bounded
    assert len(metrics.recent_latencies_ms) == 500  # but the latency window is
