'''
    Logging setup for the project.
'''

# ---------- Imports ---------- #
import os
import logging
from pythonjsonlogger import jsonlogger

# ---------- Logging setup ---------- #
def setup_logging(name: str):
    '''
        Setup logging for the project.
        
        Each module gets its own separate logger with its own handlers.
        Logs are written to logs/{name}.json
    '''
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Get a unique logger for this module (use the passed name, not __name__)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent logs from propagating to root logger (avoids duplicate logs)
    logger.propagate = False
    
    # Only add handlers if they haven't been added yet (prevents duplicate handlers on reimport)
    if not logger.handlers:
        # Terminal output
        terminal_handler = logging.StreamHandler()
        terminal_handler.setLevel(logging.INFO)
        terminal_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        terminal_handler.setFormatter(terminal_formatter)
        logger.addHandler(terminal_handler)

        # JSON file output
        log_file = f'logs/{name}.json'
        json_handler = logging.FileHandler(log_file, mode='a')
        json_handler.setLevel(logging.INFO)

        json_formatter = jsonlogger.JsonFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
            rename_fields={'asctime': 'timestamp', 'levelname': 'level', 'name': 'module'}
        )
        json_handler.setFormatter(json_formatter)
        logger.addHandler(json_handler)

        # Initial logging
        logger.info(f'Logging to {name}.json setup completed')

    return logger