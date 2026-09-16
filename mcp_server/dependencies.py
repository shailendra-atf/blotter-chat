# dependencies.py
import logging
import os
import sys
from dotenv import load_dotenv
from datetime import datetime

load_dotenv(override=True)

log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
log_file = os.path.join(
    log_dir,
    f"api_server_mcp_{datetime.now().strftime('%Y-%m-%d')}.log",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(log_file)],
)
logger = logging.getLogger("mcp-server")