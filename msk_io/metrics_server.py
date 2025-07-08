from prometheus_client import make_asgi_app


def create_metrics_app():
    """Return an ASGI app exposing Prometheus metrics."""

    return make_asgi_app()
