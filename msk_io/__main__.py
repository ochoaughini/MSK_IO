from pathlib import Path
import typer
from rich.console import Console
from .api import run_pipeline
from .config import PipelineSettings

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def run(
    dicom_dir: Path,
    config: Path,
    vault: Path,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Execute full pipeline."""
    conf = PipelineSettings(rules_path=config, verbose=verbose)
    if dry_run:
        console.print(f"[yellow]Dry run with config: {conf}")
        raise typer.Exit()
    result = run_pipeline(dicom_dir, conf, vault)
    console.print(result)


@app.command()
def load(dicom_dir: Path):
    console.print(f"Loading {dicom_dir}")


@app.command()
def segment(dicom_dir: Path):
    console.print(f"Segmenting {dicom_dir}")


if __name__ == "__main__":
    app()
