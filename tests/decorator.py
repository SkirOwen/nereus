from nereus.utils.decorator import panel_logger_decorator
import time
from nereus import logger

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