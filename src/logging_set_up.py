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
    '''
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Configure root logger without automatic handler creation
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[]
    )

    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Terminal output
    terminal_handler = logging.StreamHandler()
    terminal_handler.setLevel(logging.INFO)
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
    logger.info(f'{name} logging setup completed')

    return logger