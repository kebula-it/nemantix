import sys
import logging
import threading

from pathlib import Path

_GLOBAL_EXCEPTION_HOOK = False


def get_package_logger(
    name: str,
    level=logging.INFO,
    console_logs=True,
    log_file: Path | str | None = None,
    capture_global_exceptions=False,
) -> logging.Logger:
    global _GLOBAL_EXCEPTION_HOOK
    logger = logging.getLogger(name)

    if level not in [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ]:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s]:[%(lineno)d] %(message)s"
        )

        # package specific Handler(s)
        if log_file is not None:
            path = Path(__file__).parent.parent.parent.parent / log_file
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if console_logs:
            console_handler = logging.StreamHandler(stream=sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # logging events
        if name != "nemantix.hub.observer":
            from nemantix.hub.observer import ObserverLogHandler

            hub_handler = ObserverLogHandler()
            hub_handler.setFormatter(formatter)
            logger.addHandler(hub_handler)

        logger.propagate = False

    if (not _GLOBAL_EXCEPTION_HOOK) and capture_global_exceptions:
        _set_global_exception_hook(logger)
        _GLOBAL_EXCEPTION_HOOK = True

    return logger


def update_logger_levels(level=logging.DEBUG, prefix="nemantix"):
    prefix = str(prefix or "nemantix").lower()

    for logger_name, logger_obj in logging.root.manager.loggerDict.items():
        if isinstance(logger_obj, logging.Logger) and logger_name.startswith(prefix):
            logger_obj.setLevel(level)


def _set_global_exception_hook(logger):

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical(
            "Uncaught exception in main thread:",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    def handle_thread_exception(args):
        thread_name = args.thread.name if args.thread else "Unknown Thread"
        logger.critical(
            f"Uncaught exception in background thread '{thread_name}':",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = handle_exception
    threading.excepthook = handle_thread_exception
