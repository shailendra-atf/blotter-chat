# -------------------------------------------------------------------
# 1. Logging Configuration
# -------------------------------------------------------------------
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(f"../logs/api_server_mcp_{datetime.now().strftime('%Y-%m-%d')}.log")],
)
logger = logging.getLogger("mongodb-query-api")

