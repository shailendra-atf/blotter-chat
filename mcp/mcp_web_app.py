import sqlite3
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
import uvicorn

# 1. Initialize your core FastMCP engine
mcp = FastMCP("Web-Authoritative-DB-Server")
DB_PATH = "company_records.db"

# Expose Tool 1: Schema Discovery Registry via RAG
@mcp.tool()
def find_relevant_tables(user_intent: str) -> str:
    """
    Searches metadata registry to map the user natural text intention 
    to the literal database table target layouts.
    """
    # Lightweight text routing logic mapping keywords to target structures
    intent = user_intent.lower()
    if "revenue" in intent or "financial" in intent:
        return "MATCH FOUND: SQL Table 'financial_metrics' | Schema: (metric TEXT, value TEXT, fiscal_year INT)"
    elif "user" in intent or "profile" in intent:
        return "MATCH FOUND: MongoDB Collection 'user_profiles' | Schema: {_id, username, status}"
    return "No matching operational structures found in database catalog index."

# Expose Tool 2: Read-Only Data Engine Execution
@mcp.tool()
def execute_read_query(query_string: str) -> str:
    """Runs a read-only SELECT database operation against SQLite storage."""
    if not query_string.strip().lower().startswith("select"):
        return "Security Failure: Modification statements strictly blocked."
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query_string)
        records = cursor.fetchall()
        conn.close()
        return f"Database Output: {str(records)}"
    except Exception as e:
        return f"Runtime Database Error: {str(e)}"

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

# The SSE transport writes the POST response directly through the ASGI send
# callable, so register it as a raw route instead of wrapping it in FastAPI.
app.add_route("/messages/", sse_transport.handle_post_message, methods=["POST"])

# Quick verification root page
@app.get("/", response_class=HTMLResponse)
async def service_root():
    return "<h3>✅ MCP REST Gateway Service Active. Call endpoints at /sse and /messages</h3>"

if __name__ == "__main__":
    # Start web container framework bound to port 8000
    uvicorn.run(app, host="127.0.0.1", port=8090)
