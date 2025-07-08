from msk_io.metrics_server import create_metrics_app


def test_metrics_app():
    app = create_metrics_app()
    assert callable(app)
