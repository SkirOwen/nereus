from __future__ import annotations

import functools
from io import StringIO

import psutil
from rich.logging import RichHandler

from nereus.utils.simple_functions import convert_bytes
from nereus import logger

def memory_usage(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get memory usage before the function call
        process = psutil.Process()
        mem_before = process.memory_info().rss

        # Execute the function
        result = func(*args, **kwargs)

        # Get memory usage after the function call
        mem_after = process.memory_info().rss

        print(f"Memory used by '{func.__name__}': {convert_bytes(mem_after - mem_before)}")
        return result

    return wrapper

import logging
import time
import builtins
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.logging import RichHandler
from rich import box

from nereus import console


class PanelLogger:
    def __init__(self, title="Log Output"):
        self.console = console
        self.output = []
        self.title = title

    def log(self, message):
        self.output.append(message)

    def get_panel(self):
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        for message in self.output:
            table.add_row(message)
        return Panel(table, title=self.title, border_style="green")

def panel_logger_decorator(title="Log Output"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            console.rule("[red] test")
            panel_logger = PanelLogger(title=title)
            original_print = builtins.print
            original_logger_methods = {
                "info": logging.Logger.info,
                "warning": logging.Logger.warning,
                "error": logging.Logger.error,
                "debug": logging.Logger.debug,
            }

            def custom_print(*print_args, **print_kwargs):
                message = " ".join(map(str, print_args))
                panel_logger.log(message)
                live.update(panel_logger.get_panel())


            def custom_logger_method(method, self, message, *args, **kwargs):
                with console.capture() as capture:
                    original_logger_methods[method](self, message, *args, **kwargs)
                record = capture.get()
                print(f"{record = }")
                panel_logger.log(record)
                live.update(panel_logger.get_panel())

            # def custom_logger_method(method, self, message, *args, **kwargs):
            #     record = logging.LogRecord(
            #         name=self.name,
            #         level=logging._nameToLevel[method.upper()],
            #         pathname=self.findCaller()[0],
            #         lineno=self.findCaller()[1],
            #         msg=message,
            #         args=args,
            #         exc_info=kwargs.get('exc_info', None)
            #     )
            #     with console.capture() as capture:
            #         logger.handle(record)  # Pass the record to the logger
            #     formatted_message = capture.get()
            #     panel_logger.log(formatted_message)
            #     live.update(panel_logger.get_panel())

            builtins.print = custom_print
            for method in original_logger_methods:
                setattr(logging.Logger, method,
                        lambda self, msg, m=method, *a, **kw: custom_logger_method(m, self, msg, *a, **kw))

            # layout = Layout(name="main")
            # # layout["main"].update(panel_logger.get_panel())

            try:
                with Live(panel_logger.get_panel(), refresh_per_second=4) as live:
                    result = func(*args, **kwargs)
                    live.update(panel_logger.get_panel())
            finally:
                builtins.print = original_print
                for method in original_logger_methods:
                    setattr(logging.Logger, method, original_logger_methods[method])
                # logger.handlers = [h for h in logger.handlers if not isinstance(h, CustomRichHandler)]
            return result

        return wrapper
    return decorator


@panel_logger_decorator(title="My Function Logs")
def my_function():
    for i in range(5):
        print(f"This is a print message {i}.")
        logger.info(f"log {i}")
        time.sleep(1)


if __name__ == "__main__":
    my_function()
    print("hello")
    logger.info("test")
