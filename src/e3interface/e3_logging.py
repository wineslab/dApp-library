import logging
import os

LOG_DIR = ('.' if os.geteuid() != 0 else '') + '/logs/'

def setup_logger(name, log_file, level=logging.INFO, format_str="[%(name)s] [%(created)f] %(levelname)s - %(message)s"):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.FileHandler(log_file)
    handler.setLevel(level)

    formatter = logging.Formatter(format_str)
    handler.setFormatter(formatter)

    # Avoid adding multiple handlers if logger is reused
    if not logger.handlers:
        logger.addHandler(handler)

    return logger

e3_logger = setup_logger("e3_logger", f"{LOG_DIR}/e3.log", format_str="[E3] [%(created)f] %(levelname)s - %(message)s")
dapp_logger = setup_logger("dapp_logger", f"{LOG_DIR}/dapp.log", format_str="[dApp] [%(created)f] %(levelname)s - %(message)s")
