import os
import time
import queue
import threading
from typing import List, Callable, Any, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
from msk_io.errors import ProcessingError, ConfigurationError
from msk_io.utils.log_config import get_logger
from msk_io.utils.decorators import handle_errors

logger = get_logger(__name__)

class FileEventHandler(FileSystemEventHandler):
    def __init__(self, processing_queue: queue.Queue, supported_extensions: List[str]):
        self.processing_queue = processing_queue
        self.supported_extensions = [ext.lower() for ext in supported_extensions]
        logger.info(f"File event handler initialized. Supported extensions: {self.supported_extensions}")

    def _is_supported_file(self, event_path: str) -> bool:
        return any(event_path.lower().endswith(ext) for ext in self.supported_extensions)

    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory and self._is_supported_file(event.src_path):
            logger.info(f"Detected new file: {event.src_path}")
            self.processing_queue.put(event.src_path)
        elif event.is_directory:
            logger.debug(f"Detected new directory: {event.src_path}")

    def on_modified(self, event: FileModifiedEvent) -> None:
        if not event.is_directory and self._is_supported_file(event.src_path) and not event.src_path.endswith(('.tmp', '~')):
            logger.debug(f"Detected modified file (for potential reprocessing/completion): {event.src_path}")
            pass

class DirectoryMonitor:
    def __init__(self, config, file_processor_callback: Callable[[str], Any], supported_extensions: List[str] = ['.dcm', '.nii', '.nii.gz', '.png', '.jpg', '.pdf', '.txt']):
        self.config = config
        self.watch_directory = config.app.watch_directory
        self.file_processor_callback = file_processor_callback
        self.supported_extensions = supported_extensions
        self.processing_queue = queue.Queue()
        self.observer = Observer()
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
        logger.info(f"Directory Monitor initialized for: {self.watch_directory}")
        logger.info(f"Supported extensions: {self.supported_extensions}")
        if not os.path.isdir(self.watch_directory):
            try:
                os.makedirs(self.watch_directory, exist_ok=True)
                logger.warning(f"Watch directory '{self.watch_directory}' did not exist and was created.")
            except OSError as e:
                raise ConfigurationError(f"Watch directory '{self.watch_directory}' does not exist and cannot be created: {e}") from e

    @handle_errors
    def start_monitoring(self) -> None:
        if self.running:
            logger.warning("Directory monitor is already running.")
            return
        event_handler = FileEventHandler(self.processing_queue, self.supported_extensions)
        self.observer.schedule(event_handler, self.watch_directory, recursive=True)
        self.observer.start()
        self.running = True
        logger.info(f"Started monitoring directory: {self.watch_directory}")
        self.worker_thread = threading.Thread(target=self._process_files_from_queue, daemon=True)
        self.worker_thread.start()
        logger.info("Started file processing worker thread.")

    @handle_errors
    def stop_monitoring(self) -> None:
        if not self.running:
            logger.warning("Directory monitor is not running.")
            return
        self.observer.stop()
        self.observer.join()
        self.running = False
        if self.worker_thread:
            self.processing_queue.put(None)
            self.worker_thread.join(timeout=5)
            if self.worker_thread.is_alive():
                logger.warning("File processing worker thread did not terminate cleanly.")
        logger.info(f"Stopped monitoring directory: {self.watch_directory}")

    @handle_errors
    def _process_files_from_queue(self) -> None:
        logger.info("File processing worker thread started.")
        while True:
            file_path = self.processing_queue.get()
            if file_path is None:
                logger.info("File processing worker received stop signal.")
                break
            logger.info(f"Processing new file from queue: {file_path}")
            try:
                time.sleep(1)
                self.file_processor_callback(file_path)
                logger.info(f"Successfully processed file: {file_path}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
            finally:
                self.processing_queue.task_done()

    def __enter__(self):
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitoring()
