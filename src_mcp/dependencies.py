# dependencies.py
import logging
import os
import sys
from dotenv import load_dotenv
from datetime import datetime

load_dotenv(override=True)

logger = logging.getLogger("mcp_logger")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if not logger.handlers:
    # MCP uses stdout for JSON-RPC; diagnostics must stay off that channel.
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"api_server_mcp_{datetime.now().strftime('%Y-%m-%d')}.log",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)