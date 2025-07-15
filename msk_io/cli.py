import asyncio
import click
import os
import json
import time
import logging
from typing import Optional
from msk_io.api import MSKIOAPI
from msk_io.watch.directory_monitor import DirectoryMonitor
from msk_io.utils.log_config import get_logger, setup_logging
from msk_io.errors import MSKIOError, ConfigurationError, DataValidationError
from msk_io import CONFIG

logger = get_logger(__name__)


@click.group()
@click.option(
    "--config-file",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a custom .env configuration file.",
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Set the logging level for this run.",
)
@click.pass_context
def cli(ctx, config_file, log_level):
    try:
        if config_file:
            os.environ["MSKIO_APP_ENV_FILE"] = config_file
        from msk_io.config import load_config

        ctx.obj = {}
        ctx.obj["CONFIG"] = load_config()
        if log_level:
            ctx.obj["CONFIG"].app.log_level = log_level.upper()
            setup_logging(
                level=getattr(logging, log_level.upper()),
                log_file=ctx.obj["CONFIG"].app.log_file_path,
            )
            logger.info(f"Log level overridden to: {log_level.upper()}")
        else:
            setup_logging(
                level=getattr(logging, ctx.obj["CONFIG"].app.log_level.upper()),
                log_file=ctx.obj["CONFIG"].app.log_file_path,
            )
        logger.info("CLI initialized with configuration.")
        ctx.obj["API"] = MSKIOAPI()
    except MSKIOError as e:
        logger.critical(f"Failed to initialize CLI due to configuration error: {e}")
        click.echo(f"Error: {e}")
        ctx.exit(1)
    except Exception as e:
        logger.critical(
            f"An unexpected error occurred during CLI initialization: {e}",
            exc_info=True,
        )
        click.echo(f"An unexpected error occurred: {e}")
        ctx.exit(1)


@cli.command()
@click.argument("input_source", type=str, required=True)
@click.option(
    "--patient-id", "-p", help="Optional patient ID to associate with the data."
)
@click.option(
    "--is-url",
    "-u",
    is_flag=True,
    help="Treat INPUT_SOURCE as a remote URL to download.",
)
@click.pass_context
def process(ctx, input_source, patient_id, is_url):
    """Processes a single medical data file (local path or remote URL)."""
    api = ctx.obj["API"]

    input_file_path = None
    remote_dicom_url = None

    if is_url:
        remote_dicom_url = input_source
        logger.info(
            f"Attempting to process remote DICOM from URL: {remote_dicom_url} (Patient ID: {patient_id or 'N/A'})"
        )
    else:
        input_file_path = input_source
        if not os.path.exists(input_file_path):
            click.echo(f"Error: Local input file not found: {input_file_path}")
            ctx.exit(1)
        logger.info(
            f"Attempting to process local file: {input_file_path} (Patient ID: {patient_id or 'N/A'})"
        )
    try:
        pipeline_status = asyncio.run(
            api.process_medical_data(
                input_file_path=input_file_path,
                remote_dicom_url=remote_dicom_url,
                patient_id=patient_id,
            )
        )
        click.echo("\n--- Pipeline Summary ---")
        click.echo(f"Pipeline ID: {pipeline_status.pipeline_id}")
        click.echo(f"Status: {pipeline_status.overall_status}")
        click.echo(f"Message: {pipeline_status.overall_message}")
        if pipeline_status.final_report_path:
            click.echo(f"Final Report Path: {pipeline_status.final_report_path}")
            report = asyncio.run(
                api.get_diagnostic_report(pipeline_status.final_report_path)
            )
            if report:
                click.echo("\n--- Report Conclusion ---")
                click.echo(f"Overall Conclusion: {report.overall_conclusion}")
                click.echo(f"Number of Findings: {len(report.diagnostic_findings)}")
        else:
            click.echo("No final report generated (or path not available).")
        if pipeline_status.overall_status in ["FAILED", "COMPLETED_WITH_ERRORS"]:
            if pipeline_status.fatal_error:
                logger.error(
                    f"Fatal error details: {json.dumps(pipeline_status.fatal_error, indent=2)}"
                )
                click.echo("\n--- Fatal Error Details ---")
                click.echo(json.dumps(pipeline_status.fatal_error, indent=2))
            ctx.exit(1)
        else:
            ctx.exit(0)
    except DataValidationError as e:
        logger.error(f"Input validation failed: {e}")
        click.echo(f"Error: Input validation failed: {e}")
        ctx.exit(1)
    except MSKIOError as e:
        logger.error(f"Pipeline processing failed: {e}")
        click.echo(f"Error: Pipeline processing failed: {e}")
        ctx.exit(1)
    except Exception as e:
        logger.critical(
            f"An unexpected error occurred during processing: {e}", exc_info=True
        )
        click.echo(f"An unexpected error occurred: {e}")
        ctx.exit(1)


@cli.command()
@click.option(
    "--interval", "-i", type=int, default=5, help="Polling interval in seconds."
)
@click.pass_context
def monitor(ctx, interval):
    config = ctx.obj["CONFIG"]
    api = ctx.obj["API"]
    watch_dir = config.app.watch_directory
    if not watch_dir:
        click.echo(
            "Error: Watch directory is not configured. Please set MSKIO_APP_WATCH_DIRECTORY."
        )
        logger.error("Watch directory not configured for monitor command.")
        ctx.exit(1)
    click.echo(
        f"Monitoring directory: {watch_dir} for new files every {interval} seconds..."
    )
    click.echo("Press Ctrl+C to stop monitoring.")
    logger.info(f"Starting directory monitor on {watch_dir} with interval {interval}s.")

    @handle_errors
    def process_new_file_callback(file_path: str):
        logger.info(
            f"Callback triggered for new file: {file_path}. Initiating pipeline processing."
        )
        click.echo(f"\n[Detected] New file: {file_path}. Starting pipeline...")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            pipeline_status = loop.run_until_complete(
                api.process_medical_data(input_file_path=file_path)
            )
            loop.close()
            click.echo(
                f"Pipeline for {os.path.basename(file_path)} completed with status: {pipeline_status.overall_status}"
            )
            if pipeline_status.final_report_path:
                click.echo(f"Report: {pipeline_status.final_report_path}")
            if pipeline_status.overall_status in ["FAILED", "COMPLETED_WITH_ERRORS"]:
                click.echo(f"Details: {pipeline_status.overall_message}")
                if pipeline_status.fatal_error:
                    click.echo(
                        f"Error: {json.dumps(pipeline_status.fatal_error, indent=2)}"
                    )
        except Exception as e:
            logger.error(
                f"Error during automated processing of {file_path}: {e}", exc_info=True
            )
            click.echo(f"Error processing {file_path}: {e}")

    monitor_instance = DirectoryMonitor(config, process_new_file_callback)
    try:
        monitor_instance.start_monitoring()
        while True:
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\nStopping monitoring.")
        logger.info("Directory monitor interrupted by user.")
    finally:
        monitor_instance.stop_monitoring()
        click.echo("Monitoring stopped.")
        logger.info("Directory monitor gracefully stopped.")
    ctx.exit(0)


@cli.command()
@click.option("--pipeline-id", help="Optional: Specific pipeline ID to check.")
@click.pass_context
def status(ctx, pipeline_id):
    api = ctx.obj["API"]
    if pipeline_id:
        logger.info(f"Checking status for pipeline ID: {pipeline_id}")
        try:
            pipeline_status = asyncio.run(api.get_pipeline_status(pipeline_id))
            if pipeline_status:
                click.echo(
                    f"--- Pipeline Status for ID: {pipeline_status.pipeline_id} ---"
                )
                click.echo(f"Overall Status: {pipeline_status.overall_status}")
                click.echo(f"Message: {pipeline_status.overall_message}")
                click.echo(f"Start Time: {pipeline_status.start_time}")
                if pipeline_status.end_time:
                    click.echo(f"End Time: {pipeline_status.end_time}")
                    click.echo(
                        f"Duration: {pipeline_status.total_duration_seconds:.2f} seconds"
                    )
                for task in pipeline_status.tasks_status:
                    click.echo(f"  - Task '{task.task_name}': {task.status}")
                    if task.error_details:
                        click.echo(f"    Error: {json.dumps(task.error_details)}")
                if pipeline_status.final_report_path:
                    click.echo(f"Final Report: {pipeline_status.final_report_path}")
            else:
                click.echo(
                    f"Pipeline with ID '{pipeline_id}' not found or status not available."
                )
                logger.warning(f"Status check: Pipeline ID '{pipeline_id}' not found.")
        except MSKIOError as e:
            logger.error(f"Failed to retrieve status for {pipeline_id}: {e}")
            click.echo(f"Error: Failed to retrieve status: {e}")
            ctx.exit(1)
        except Exception as e:
            logger.critical(
                f"An unexpected error occurred during status check: {e}", exc_info=True
            )
            click.echo(f"An unexpected error occurred: {e}")
            ctx.exit(1)
    else:
        click.echo("This command conceptually shows recent pipeline statuses.")
        click.echo("Please provide a --pipeline-id to check a specific run.")
        logger.info("Status command called without a specific pipeline ID.")
        click.echo("Example: msk-io status --pipeline-id <UUID>")
    ctx.exit(0)


@cli.command()
@click.pass_context
def config(ctx):
    logger.info("Displaying current configuration.")
    config_obj = ctx.obj["CONFIG"]
    config_dict = config_obj.model_dump(mode="json", exclude_sensitive=True)
    click.echo(json.dumps(config_dict, indent=2))
    ctx.exit(0)


if __name__ == "__main__":
    cli(obj={})
