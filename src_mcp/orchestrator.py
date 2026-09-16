import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
load_dotenv(override=True)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from src_mcp.context import QUERY_GUARDRAILS_CONTEXT, QUERY_SYSTEM_PROMPT
from src_mcp.config import COLLECTION1, SIMILARITY_THRESHOLD, MCP_SERVER_SSE_URL
from src_mcp.models import AgentState, TaskState
from src_mcp.dependencies import logger
from src_mcp.utils import convert_messages_to_string, get_unique_names_from_database
from src_mcp.rag_ingestion import initialize_chromadb

async def _bootstrap_runtime():
    UNIQUE_NAMES_DICT = await get_unique_names_from_database()
    chroma_collection = initialize_chromadb()
    return chroma_collection, UNIQUE_NAMES_DICT

# ==========================================
# UPDATED BUILD ORCHESTRATOR (USING SSE API)
# ==========================================
async def build_orchestrator():
    chroma_collection, UNIQUE_NAMES_DICT = await _bootstrap_runtime()
    
    # ---------------------------------------------------------------
    # CHANGED: Replaced stdio command config with SSE HTTP Endpoint API
    # ---------------------------------------------------------------
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
        # logger.info(f"{convert_messages_to_string(state.get("messages",""))=}")
        state["messages"] = [msg.content for msg in state.get("messages", "")]
        # results = chroma_collection.query(query_texts=[latest_user_prompt], n_results=min(2, chroma_collection.count()))
        results = chroma_collection.query(query_texts=state["messages"], n_results=min(2, chroma_collection.count()))
        if not results or not results.get("metadatas") or not results["metadatas"][0]:
            return {
                "retrieved_docs": [],
                "final_response": "No matching database schemas located for this intent."
            }
        
        distances = results["distances"][0]
        indices_sim_scores = [idx for idx, d in enumerate(distances) if 1-d > SIMILARITY_THRESHOLD]
        logger.info(f"similarity_scores: {[1-d for idx, d in enumerate(distances)]}, {[1-d for idx, d in enumerate(distances) if 1-d > SIMILARITY_THRESHOLD]=}")
        retrieved_docs = [metadata for idx, metadata in enumerate(results["metadatas"][0]) if idx in indices_sim_scores]
        state["retrieved_docs"] = retrieved_docs
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
                    "messages": convert_messages_to_string(state["messages"])
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
        # logger.info(f"task state messages: {state["messages"]}")
        result = await mcp_generate_query.ainvoke({
            "system_prompt": query_system_prompt,
            "db_type": doc.get("db_type",""),
            "target_resource": doc.get("db",""),
            "schema_info": doc.get("schema",""),
            # "user_intent": state.get("messages", "")[-1].content,
            "user_intent": state["messages"],
        })

        logger.info(f'{result=}')
        
        if isinstance(result, dict):
            query_str = result["text"]
        elif isinstance(result, list):
            query_str = json.loads(result[0]["text"].replace('```json','').replace('```','')).get("query","")
            query_str = query_str[query_str.find('['):]
        else:
            query_str = result[result.find('['):]
        logger.info(f'{query_str=}')
        return {"mongodb_queries": [query_str]}

    async def execute_mongodb_query_node(state: AgentState):
        """Executes read operations against the database via the MCP API."""
        logger.info(f"execute_mongodb_query_node called")
        results = []
        
        for raw_input in state.get("mongodb_queries", []):
            if mcp_exec_read:
                # Call execute_read_query API tool endpoint
                logger.info(f"{raw_input=}")
                query_str = raw_input["text"] if isinstance(raw_input, dict) else raw_input
                # query_str = """[{"$match": {"TraderName": "Rajesh Mahadevan", "ValuationDate": {"$gte": {"$date": "2026-01-01T00:00:00Z"}, "$lte": {"$date": "2026-01-31T23:59:59Z"}}}}, {"$group": {"_id": null, "MTDPnL": {"$sum": "$MTDPnL"}}}, {"$project": {"_id": 0, "MTDPnL": 1}}]"""
                res = await mcp_exec_read.ainvoke({"query_string": query_str})
                
                logger.info(f"{json.loads(res[0]['text'])=}")
                # if bedrock:
                #     json.loads(res[0]['text'])['query_results']
                res = json.loads(res[0]['text'])['query_results']
                results.append(res)

        state["query_results"] = results
        logger.info(f"{state=}")
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

    return builder.compile()
