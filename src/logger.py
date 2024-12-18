import logging
import os
from datetime import datetime

def get_logger():
    # Define the log filename and directory
    LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Full path to the log file
    LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

    # Configure logging
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger()
