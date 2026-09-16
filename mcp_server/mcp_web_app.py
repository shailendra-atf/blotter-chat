import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from bson.json_util import dumps, loads
from pymongo.errors import ExecutionTimeout, PyMongoError
import sqlite3
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_MCP_ROOT = PROJECT_ROOT / "src_mcp"
for path in (str(PROJECT_ROOT), str(SRC_MCP_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src_mcp.config import LIMIT_RESULTS, DB_NAME1, COLLECTION1, MAX_RESULTS
from mcp_server.dependencies import logger
from src_mcp.utils import validate_pipeline_stages, startup_db_client, initialize_llm_client

# Unicode-compatible JSON helpers are not required by this minimal web MCP server.
# Keeping imports minimal avoids the `bson` module dependency in environments where PyMongo is missing.

load_dotenv(override=True)

server_llm = initialize_llm_client()

# 1. Initialize your core FastMCP engine
mcp = FastMCP("Web-Authoritative-DB-Server")

@mcp.tool()
async def generate_mongodb_query(
    system_prompt: str,
    db_type: str,
    target_resource: str,
    schema_info: str,
    user_intent: str,
) -> dict:
    """
    Translates a plain-text user intent into a valid, executable SQL query string
    or MongoDB dictionary filter based on the target resource and its schema.
    """
    try:
        user_intent = user_intent

        db_type_upper = db_type.upper()

        if db_type_upper == "MONGODB":
            if server_llm is None:
                return {"response": "LLM client not initialized."}
            # logger.info(f"{db_type_upper=}, {system_prompt=}, {user_intent=}, {target_resource=}")
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", """Intent: {user_intent}. schema: {schema_info}""")
            ])
            # logger.info(f"{prompt_template=}")
            chain = prompt_template | server_llm
            response = await asyncio.wait_for(
                chain.ainvoke({
                    "user_intent": user_intent,
                    "target_resource": target_resource,
                    "schema_info": schema_info,
                }),
                timeout=180,
            )
            # logger.info(f"{prompt_template=}")
        content = response.content if hasattr(response, "content") else response
        logger.info(f"{content=}")
        return json.dumps({"query": content})
            
    except asyncio.TimeoutError:
        logger.error("Timed out while generating the MongoDB query from the LLM.")
        return {"response": "LLM query generation timed out after 180 seconds."}
    except Exception as e:
        return {"response": f"Error Executing mongodb query generation tool {e}"}

# Expose Tool 2: Read-Only Data Engine Execution
@mcp.tool()
async def execute_read_query(query_string: str, collection: str = COLLECTION1) -> Dict[str, Any]:
    """Runs a read-only SELECT database operation against SQLite storage."""
    db = await startup_db_client()
    if db is None:
        logger.error("Database handle is unavailable.")
        return {"query_results":"Database connection error."}

    if not query_string:
        logger.error("No mongo query provided.")
        return {"query_results":"No mongo query is found."}

    try:
        logger.info(f"Validating and executing async aggregation pipeline on collections.")

        pipeline = loads(query_string)
        logger.info(f"{pipeline=}")
        validation_error = validate_pipeline_stages(pipeline)
        if validation_error:
            logger.error(f"Pipeline rejected: {validation_error}")
            return {"query_results":f"Pipeline Validation Error: {validation_error}"}
        # logger.info(f"{pipeline=}")

        target_collection = db[collection or COLLECTION1]
        cursor = target_collection.aggregate(pipeline, maxTimeMS=120000)
        mongo_results = await cursor.to_list()

        result_count = len(mongo_results)
        if result_count > MAX_RESULTS:
            return {"query_results":(
                f"Query returned {result_count} results, which exceeds the maximum limit of {MAX_RESULTS}. "
                "There are too many results. Please refine your query by getting sumn over the group."
            )}
        elif result_count == 0:
            return {"query_results":(
                "No matching PnL records were found for the specified criteria."
            )}
        # logger.info(f"{pipeline=}")

        if not mongo_results:
            logger.info("Query returned no matching records.")
            return {"query_results":"No matching PnL records were found for the specified criteria."}
        logger.info(f"{mongo_results=}")

        return {"query_results": mongo_results}

    except ExecutionTimeout as e:
        logger.error(f"MongoDB Execution Timeout (maxTimeMS expired): {str(e)}")
        return {"query_results":"Query Timeout: The requested aggregation took too long to execute. Please narrow down your date range or filter criteria."}

    except PyMongoError as e:
        logger.error(f"MongoDB Driver Error: {str(e)}", exc_info=True)
        return {"query_results":f"Database Query Error: {str(e)}"}

    except json.JSONDecodeError as e:
        logger.error(f"JSON Parsing Error: {str(e)}")
        return {"query_results":f"JSON Parsing Error: Invalid MongoDB query payload. Details: {str(e)}"}

    except Exception as e:
        logger.error(f"Execution Error: {str(e)}", exc_info=True)
        return {"query_results":f"Query Execution Error: {str(e)}"}

# 2. Build standard FastAPI routing framework
app = FastAPI(title="MCP HTTP Gateway API")

# Setup the Server-Sent Events Transport bridge hook
sse_transport = SseServerTransport("/messages/")

# 3. Expose the MCP Protocol endpoints to the web topology
@app.get("/sse")
async def handle_sse_endpoint(request: Request):
    """Establishes the persistent SSE connection stream for the client connection handshake."""
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await mcp._mcp_server.run(
            streams[0],
            streams[1],
            mcp._mcp_server.create_initialization_options(),
        )
    return Response()

# The SSE transport is a raw ASGI app, so mount it instead of wrapping it in
# FastAPI's request-handler adapter.
app.mount("/messages/", sse_transport.handle_post_message)

# Quick verification root page
@app.get("/", response_class=HTMLResponse)
async def service_root():
    return "<h3>✅ MCP REST Gateway Service Active. Call endpoints at /sse and /messages</h3>"

if __name__ == "__main__":
    # Start web container framework bound to port 8000
    uvicorn.run(app, host="127.0.0.1", port=8090)
