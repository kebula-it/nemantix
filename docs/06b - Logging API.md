# Nemantix Logging API

The Nemantix logging utility (`nemantix.common.logger`) provides a centralized way to manage logging, supporting console
output, file writing, event hub integration, and global exception handling.

## 1. Core API Functions

* **`get_package_logger`**: Creates and configures loggers for individual modules. It applies standard formatting and
  handles routing logs to the console, a specified file, or the internal event hub. Parameters:
  * **`name`** (`str`): The name of the logger (typically `__name__` of the calling module).
  * **`level`** (`int`): The logging level. Defaults to `logging.INFO`. Fallbacks to `INFO` if an invalid level is
    provided.
  * **`console_logs`** (`bool`): If `True`, attaches a `StreamHandler` routing logs to `sys.stdout`.
  * **`log_file`** (`Path | str | None`): If provided, attaches a `FileHandler` that writes to the specified path (
    relative to the package root).
  * **`capture_global_exceptions`** (`bool`): If `True`, sets global hooks (`sys.excepthook` and `threading.excepthook`)
    to capture and log unhandled exceptions using this logger.

* **`update_logger_levels`**: Dynamically adjusts the logging level (e.g., `logging.DEBUG`, `logging.INFO`) for all
  initialized loggers under a specified namespace prefix.
  * `level` (`int`): The new target logging level (e.g., `logging.DEBUG`).
  * `prefix` (`str`): The namespace prefix to target. Defaults to `"nemantix"`.
* **`disable_console_logs`**: Iterates through loggers under a namespace prefix and safely removes console output
  handlers (`StreamHandler`) without disrupting file logs or hub events.
  * `prefix` (`str`): The namespace prefix to target. Defaults to `"nemantix"`.

## 2. Practical Usage in `Agent`

### Dynamic Configuration and Log Disabling

The `Agent` class constructor dynamically configures logging behavior based on the `log_level` parameter provided during
initialization.

* **Updating Levels:** If a valid logging level is passed (e.g., `logging.DEBUG`), the agent logs the intent to update
  the level and calls `update_logger_levels(level=log_level)` to apply it package-wide.
* **Disabling Console Output:** If the `log_level` argument is passed as the string `'disable'`, the agent invokes
  `disable_console_logs()` to turn off all the module-level console loggers of Nemantix.

```python
# example usage
import logging
from nemantix.core import Agent

# sets the Nemantix modules log level to WARNING
agent = Agent(..., log_level=logging.WARNING)

# or, to turn off all internal console logs
agent = Agent(..., log_level='disable')
```
