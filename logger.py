import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger():
    """
    Sets up a centralized logger for the application.
    
    - Creates a 'logs' directory if it doesn't exist.
    - Logs messages to 'logs/app.log'.
    - Implements log rotation to keep log files from growing too large.
    """
    # --- Create logs directory ---
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # --- Get the logger ---
    # Using a named logger avoids conflicts with other libraries' loggers
    logger = logging.getLogger("energy_forecaster_app")
    
    # --- Prevent adding multiple handlers ---
    # This is important for Streamlit's execution model
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # --- Create a formatter ---
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
    )

    # --- Create a file handler with rotation ---
    log_file = os.path.join(log_dir, 'app.log')
    # Rotates the log file when it reaches 2MB, keeps 5 backup files
    handler = RotatingFileHandler(log_file, maxBytes=2*1024*1024, backupCount=5)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    # --- Add handler to the logger ---
    logger.addHandler(handler)

    return logger

# Initialize and get the logger instance
logger = setup_logger()
