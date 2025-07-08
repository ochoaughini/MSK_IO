from typer.testing import CliRunner
from msk_io.cli import app


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert "Usage" in result.stdout
