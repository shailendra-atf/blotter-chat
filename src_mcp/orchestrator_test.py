import asyncio
import os
import sys
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv(override=True)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pathlib import Path

from src_mcp.context import QUERY_GUARDRAILS_CONTEXT, QUERY_SYSTEM_PROMPT
from src_mcp.config import COLLECTION1, SIMILARITY_THRESHOLD
from src_mcp.models import AgentState, TaskState

# import src_mcp.dependencies # this statement is required
from src_mcp.dependencies import logger
from src_mcp.utils import startup_db_client, get_unique_names_from_database
from src_mcp.rag_ingestion import initialize_chromadb

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Configure the endpoint for your HTTP SSE MCP Server
MCP_SERVER_SSE_URL = "http://127.0.0.1:8090/sse"

async def _bootstrap_runtime():
    UNIQUE_NAMES_DICT = await get_unique_names_from_database()
    chroma_collection = initialize_chromadb()
    return chroma_collection, UNIQUE_NAMES_DICT


# ==========================================
# UPDATED BUILD ORCHESTRATOR (USING SSE API)
# ==========================================
async def build_orchestrator():
    chroma_collection, UNIQUE_NAMES_DICT = await _bootstrap_runtime()
    
    mcp_client = MultiServerMCPClient({
        "mcp_web_tools": {
            "url": MCP_SERVER_SSE_URL,
            "transport": "sse"
        }
    })

    # Load MCP Tools dynamically via the HTTP/SSE connection
    tools_list = await mcp_client.get_tools()
    mcp_tools = {tool.name: tool for tool in tools_list}
    
    # Fetch HTTP API Tools exposed by FastMCP
    mcp_generate_query = mcp_tools.get("generate_mongodb_query")
    mcp_exec_read = mcp_tools.get("execute_read_query")

    # ==========================================
    # Graph Nodes Calling MCP Tools via API
    # ==========================================

    async def rag_worker(state: AgentState):
        """RAG retriever node enhanced with MCP schema lookup API."""
        logger.info(f"rag_worker_node called")
        
        # Call RAG tool exposed over MCP SSE API
        results = chroma_collection.query(query_texts=[state.get("messages")], n_results=min(2, chroma_collection.count()), include=["metadatas","distances"])
        if not results or not results.get("metadatas") or not results["metadatas"][0]:
            return {
                "retrieved_docs": [],
                "final_response": "No matching database schemas located for this intent."
            }
        distances = results["distances"][0]
        indices_sim_scores = [idx for idx, d in enumerate(distances) if 1-d > SIMILARITY_THRESHOLD]
        logger.info(f"similarity_scores: {[1-d for idx, d in enumerate(distances)]}, {[1-d for idx, d in enumerate(distances) if 1-d > SIMILARITY_THRESHOLD]=}")
        retrieved_docs = [metadata for idx, metadata in enumerate(results["metadatas"][0]) if idx in indices_sim_scores]
        return {"retrieved_docs": retrieved_docs}

    def supervisor_fanout(state: AgentState):
        """Supervisor router: Dynamic parallel fan-out using Send()."""
        logger.info(f"supervisor_fanout called")
        fanout_branches = []
        for doc in state["retrieved_docs"]:
            logger.info(f'{doc=}')
            fanout_branches.append(
                Send("generate_mongodb_query_node", {
                    "doc": doc, 
                    "messages": state["messages"]
                })
            )
        return fanout_branches

    async def generate_mongodb_query_node(state: TaskState):
        """Node formulating target queries."""
        logger.info(f"generate_mongodb_query_node called")


        doc = state.get("doc", {})

        query_system_prompt = QUERY_SYSTEM_PROMPT.format(
            QUERY_GUARDRAILS_CONTEXT=QUERY_GUARDRAILS_CONTEXT,
            UNIQUE_NAMES_DICT=UNIQUE_NAMES_DICT,
            CONTEXT=doc.get("schema", "").format(COLLECTION1=COLLECTION1),
        )
        result = await mcp_generate_query.ainvoke({
            "system_prompt": query_system_prompt,
            "db_type": doc.get("db_type",""),
            "target_resource": doc.get("db",""),
            "schema_info": doc.get("schema",""),
            "user_intent": state.get("messages", ""),#[-1].content,
        })

        logger.info(f'{result=}')
        if isinstance(result, dict):
            query_str = result["text"]
        elif isinstance(result, list):
            query_str = json.loads(result[0]["text"].replace('```json','').replace('```','')).get("query","")
        else:
            query_str = result
        logger.info(f'{query_str=}')
        return {"mongodb_queries": [query_str]}

    async def execute_mongodb_query_node(state: AgentState):
        """Executes read operations against the database via the MCP API."""
        logger.info(f"execute_mongodb_query_node called")
        results = []
        
        for raw_input in state.get("mongodb_queries", []):
            if mcp_exec_read:
                # Call execute_read_query API tool endpoint
                query_str = raw_input["text"] if isinstance(raw_input, dict) else raw_input
                query_str = """[{"$match": {"TraderName": "Rajesh Mahadevan", "ValuationDate": {"$gte": {"$date": "2026-01-01T00:00:00Z"}, "$lte": {"$date": "2026-01-31T23:59:59Z"}}}}, {"$group": {"_id": null, "MTDPnL": {"$sum": "$MTDPnL"}}}, {"$project": {"_id": 0, "MTDPnL": 1}}]"""
                res = await mcp_exec_read.ainvoke({"query_string": query_str})
                logger.info(f"{json.loads(res[0]['text'])=}")
                res = json.loads(res[0]['text'])['mongo_results']
                results.append(res)
                
        return {"query_results": results}

    # ==========================================
    # Graph Construction
    # ==========================================
    builder = StateGraph(AgentState)

    builder.add_node("rag_worker", rag_worker)
    builder.add_node("generate_mongodb_query_node", generate_mongodb_query_node)
    builder.add_node("execute_mongodb_query_node", execute_mongodb_query_node)

    # Graph Connections
    builder.add_edge(START, "rag_worker")

    builder.add_conditional_edges(
        "rag_worker",
        supervisor_fanout,
        ["generate_mongodb_query_node"]
    )
    # builder.add_edge("generate_mongodb_query_node", END)
    builder.add_edge("generate_mongodb_query_node", "execute_mongodb_query_node")
    builder.add_edge("execute_mongodb_query_node", END)

    app = builder.compile()

    # with open("langgraph_architecture.png", "wb") as f:
    #     f.write(app.get_graph().draw_mermaid_png())
    return app

async def run_pipeline():
    cross_db_query = "Show me monthly MTDPnL of Rajesh Mahadevan starting for 2026, consider last day of the month"
    # cross_db_query = "show me monthly limits OF Rajesh Mahadevan starting for 2026, consider last day of the month"
    # cross_db_query = "Show me monthly MTDPnL and limits of Rajesh Mahadevan starting for 2026, consider last day of the month"

    initial_state = {
        "messages": cross_db_query,
        "retrieved_docs": [],
        "mongodb_queries": [],
        "query_results": [],
        "final_response": ""
    }

    graph = await build_orchestrator()
    final_state = await graph.ainvoke(initial_state)
    logger.info(f"Pipeline Result: {final_state}")

if __name__ == "__main__":
    asyncio.run(run_pipeline())