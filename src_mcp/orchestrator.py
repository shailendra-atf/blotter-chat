import json
from typing import Any, Dict, List
import pandas as pd
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

load_dotenv(override=True)

from src_mcp.config import COLLECTION1, MCP_SERVER_SSE_URL, SIMILARITY_THRESHOLD
from src_mcp.context import QUERY_GUARDRAILS_CONTEXT, QUERY_SYSTEM_PROMPT
from src_mcp.dependencies import logger
from src_mcp.models import AgentState, TaskState
from src_mcp.rag_ingestion import initialize_chromadb
from src_mcp.utils import convert_messages_to_string, get_unique_names_from_database, initialize_llm_client


async def _bootstrap_runtime():
    unique_names_dict = await get_unique_names_from_database()
    chroma_collection = initialize_chromadb()
    return chroma_collection, unique_names_dict


def _extract_query_text(response: Any) -> str:
    if hasattr(response, "content"):
        response = response.content

    if isinstance(response, str):
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1]).strip()
        if cleaned.startswith("{") or cleaned.startswith("["):
            try:
                payload = json.loads(cleaned)
                if isinstance(payload, dict):
                    if "query" in payload:
                        return _extract_query_text(payload["query"])
                    if "text" in payload:
                        return _extract_query_text(payload["text"])
                elif isinstance(payload, list):
                    return json.dumps(payload)
            except json.JSONDecodeError:
                pass

        # Some model responses include an explanation before the JSON pipeline.
        decoder = json.JSONDecoder()
        for index, character in enumerate(cleaned):
            if character != "[":
                continue
            try:
                payload, _ = decoder.raw_decode(cleaned[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, list):
                return json.dumps(payload)
        return cleaned

    if isinstance(response, dict):
        if "query" in response:
            return _extract_query_text(response["query"])
        if "text" in response:
            return _extract_query_text(response["text"])
        if "response" in response:
            return _extract_query_text(response["response"])

    if isinstance(response, list):
        if not response:
            return "[]"
        first = response[0]
        if isinstance(first, dict) and "text" in first:
            return _extract_query_text(first["text"])
        if all(isinstance(item, dict) for item in response):
            return json.dumps(response)

    return str(response)


def _normalize_mongodb_pipeline(query_text: str) -> str | None:
    """Return a JSON MongoDB aggregation pipeline or None for model/tool errors."""
    # if not isinstance(query_text, list):
    #     logger.info(query_text[0])
    #     query_text = json.loads(query_text[0].get('text','')).get("query","")
    if not isinstance(query_text, str):
        return None

    cleaned = query_text.strip()
    if not cleaned or cleaned.startswith(("Error ", "error ", "I couldn't", "I could not")):
        return None

    try:
        pipeline = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if not isinstance(pipeline, list) or not all(isinstance(stage, dict) for stage in pipeline):
        return None

    return json.dumps(pipeline)


def _extract_query_results(response: Any) -> List[Dict[str, Any]]:
    if isinstance(response, str):
        try:
            return _extract_query_results(json.loads(response))
        except json.JSONDecodeError:
            return []

    if isinstance(response, dict):
        query_results = response.get("query_results", response.get("result", response.get("data")))
        if isinstance(query_results, list):
            return query_results
        if isinstance(query_results, dict):
            return [query_results]
        if isinstance(query_results, str):
            try:
                return _extract_query_results(json.loads(query_results))
            except json.JSONDecodeError:
                return []

    if isinstance(response, list):
        if response and isinstance(response[0], dict) and "text" in response[0]:
            return _extract_query_results(response[0]["text"])
        if response and isinstance(response[0], dict) and "query_results" in response[0]:
            merged: List[Dict[str, Any]] = []
            for item in response:
                if isinstance(item, dict) and isinstance(item.get("query_results"), list):
                    merged.extend(item["query_results"])
            return merged
        return response

    return []


async def build_orchestrator():
    chroma_collection, unique_names_dict = await _bootstrap_runtime()
    analysis_llm = initialize_llm_client()

    mcp_client = MultiServerMCPClient({
        "mcp_web_tools": {
            "url": MCP_SERVER_SSE_URL,
            "transport": "sse",
        }
    })

    tools_list = await mcp_client.get_tools()
    mcp_tools = {tool.name: tool for tool in tools_list}

    mcp_generate_query = mcp_tools.get("generate_mongodb_query")
    mcp_exec_read = mcp_tools.get("execute_read_query")

    if mcp_generate_query is None or mcp_exec_read is None:
        raise RuntimeError("Required MCP tools are unavailable: generate_mongodb_query and execute_read_query")

    async def rag_worker(state: AgentState):
        logger.info("rag_worker called")
        user_intent = convert_messages_to_string(state.get("messages", []))
        if not user_intent.strip():
            return {"retrieved_docs": [], "final_response": "I couldn't understand the request."}

        try:
            total_docs = chroma_collection.count()
            if total_docs <= 0:
                return {"retrieved_docs": [], "final_response": "No database schemas are available for this request."}
            query_limit = min(3, total_docs)
            results = chroma_collection.query(query_texts=[user_intent], n_results=query_limit)
        except Exception:
            logger.exception("RAG schema lookup failed")
            return {"retrieved_docs": [], "final_response": "I couldn't find relevant database schemas for this request."}

        if not results or not results.get("metadatas") or not results["metadatas"][0]:
            return {"retrieved_docs": [], "final_response": "No matching database schemas located for this intent."}

        distances = results.get("distances", [[]])[0]
        indices = [idx for idx, distance in enumerate(distances) if 1 - distance > SIMILARITY_THRESHOLD]
        similarity_scores = [1-distance for idx, distance in enumerate(distances)]
        logger.info(f"{similarity_scores=}")
        retrieved_docs = [
            metadata for idx, metadata in enumerate(results["metadatas"][0]) if idx in indices
        ]

        if not retrieved_docs:
            return {"retrieved_docs": [], "final_response": "No matching database schemas located for this intent."}

        return {"retrieved_docs": retrieved_docs}

    def supervisor_fanout(state: AgentState):
        logger.info("supervisor_fanout called")
        user_intent = convert_messages_to_string(state.get("messages", []))
        branches: List[Send] = []
        for doc in state.get("retrieved_docs", []):
            logger.info(f"{doc=}")
            branches.append(
                Send(
                    "generate_mongodb_query_node",
                    {
                        "doc": doc,
                        "user_intent": user_intent,
                    },
                )
            )
        return branches

    async def generate_mongodb_query_node(state: TaskState):
        logger.info("generate_mongodb_query_node called")
        doc = state.get("doc", {})
        user_intent = state.get("user_intent", "")

        schema_context = doc.get("schema", "")
        collection = doc.get("target", "")
        formatted_context = (
            schema_context.replace("{COLLECTION}", collection)
            if schema_context
            else ""
        )

        query_system_prompt = QUERY_SYSTEM_PROMPT.format(
            QUERY_GUARDRAILS_CONTEXT=QUERY_GUARDRAILS_CONTEXT,
            UNIQUE_NAMES_DICT=unique_names_dict,
        )

        try:
            result = await mcp_generate_query.ainvoke({
                "system_prompt": query_system_prompt,
                "db_type": doc.get("db_type", "MONGODB"),
                "target_resource": doc.get("db", ""),
                "schema_info": formatted_context,
                "user_intent": user_intent,
            })
            query_text = _normalize_mongodb_pipeline(_extract_query_text(result))
            if not query_text or query_text == "[]":
                logger.error("MCP query generation returned no valid MongoDB pipeline: %r", result)
                return {
                    "mongodb_queries": [],
                    "final_response": "I couldn't generate a valid database query for this request.",
                }
            logger.info(f"{query_text=}")
            return {
                "mongodb_queries": [{
                    "query": query_text,
                    "collection": collection,
                }]
            }
        except Exception:
            logger.exception("Failed to generate MongoDB query via MCP")
            return {"final_response": "I couldn't generate the database query for this request."}

    async def execute_mongodb_query_node(state: AgentState):
        logger.info("execute_mongodb_query_node called")
        results: List[Dict[str, Any]] = []

        for raw_input in state.get("mongodb_queries", []):
            query_string = raw_input
            if isinstance(raw_input, dict):
                query_string = raw_input.get("query", raw_input)
                collection = raw_input.get("collection")
            if not isinstance(query_string, str):
                query_string = json.dumps(query_string)

            query_string = _normalize_mongodb_pipeline(query_string)
            if not query_string:
                logger.error("Skipping invalid MongoDB pipeline: %r", raw_input)
                return {
                    "query_results": [],
                    "final_response": "I couldn't generate a valid database query for this request.",
                }

            try:
                logger.info("Executing MongoDB query via MCP for collection %s", collection)
                response = await mcp_exec_read.ainvoke({
                    "query_string": query_string,
                    "collection": collection,
                })
                extracted = _extract_query_results(response)
                if extracted:
                    results.extend(extracted)
            except Exception as exc:
                logger.exception("MongoDB query execution failed via MCP")
                return {"query_results": [], "final_response": "I couldn't retrieve the requested data right now."}

        logger.info(f"{results=}")
        return {"query_results": results}

    async def format_response_node(state: AgentState):
        logger.info("format_response_node called")
        query_results = state.get("query_results", [])
        existing_response = state.get("final_response", "")

        if not query_results:
            return {
                "final_response": existing_response
                or "No matching records found in the database for the given criteria."
            }

        try:
            dataframe = pd.DataFrame(query_results)
            if dataframe.empty:
                return {
                    "final_response": existing_response
                    or "No matching records found in the database for the given criteria."
                }

            # markdown_table = dataframe.to_markdown(index=False, tablefmt="github")
            analysis = "The query returned data, but an analysis could not be generated."
            if analysis_llm is not None:
                user_intent = convert_messages_to_string(state.get("messages", []))
                analysis_prompt = (
                    f"Create a Markdown table from the Results JSON:\n{json.dumps(query_results, default=str)}"
                    f"Also generate deep analysis on this data for the user's intent: {user_intent}"
                )
                response = await analysis_llm.ainvoke(analysis_prompt)
                content = response.content if hasattr(response, "content") else response
                if isinstance(content, list):
                    content = "\n".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("text")
                    )
                if str(content).strip():
                    analysis = str(content).strip()

            return {"final_response": f"## Results\n\n{analysis}"}
        except Exception:
            logger.exception("Failed to format or analyze query results")
            return {
                "final_response": "I couldn't format the query results into a readable table, but the data was retrieved successfully."
            }

    async def format_response_node1(state: AgentState):
        logger.info("format_response_node called")
        query_results = state.get("query_results", [])
        existing_response = state.get("final_response", "")

        if not query_results:
            return {
                "final_response": existing_response
                or "No matching records found in the database for the given criteria."
            }

        try:
            dataframe = pd.DataFrame(query_results)
            if dataframe.empty:
                return {
                    "final_response": existing_response
                    or "No matching records found in the database for the given criteria."
                }

            markdown_table = dataframe.to_markdown(index=False, tablefmt="github")
            analysis = "The query returned data, but an analysis could not be generated."
            if analysis_llm is not None:
                user_intent = convert_messages_to_string(state.get("messages", []))
                analysis_prompt = (
                    "Analyze the database results below for the user's request. "
                    "Use only facts supported by the results. Mention important trends, "
                    "comparisons, totals, or exceptions when present. Do not repeat the "
                    "table, do not write markdown headings, and keep the analysis concise.\n\n"
                    f"User request:\n{user_intent}\n\n"
                    f"Results JSON:\n{json.dumps(query_results, default=str)}"
                )
                response = await analysis_llm.ainvoke(analysis_prompt)
                content = response.content if hasattr(response, "content") else response
                if isinstance(content, list):
                    content = "\n".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("text")
                    )
                if str(content).strip():
                    analysis = str(content).strip()

            return {"final_response": f"## Results\n\n{markdown_table}\n\n## Analysis\n\n{analysis}"}
        except Exception:
            logger.exception("Failed to format or analyze query results")
            return {
                "final_response": "I couldn't format the query results into a readable table, but the data was retrieved successfully."
            }

    builder = StateGraph(AgentState)
    builder.add_node("rag_worker", rag_worker)
    builder.add_node("generate_mongodb_query_node", generate_mongodb_query_node)
    builder.add_node("execute_mongodb_query_node", execute_mongodb_query_node)
    builder.add_node("format_response_node", format_response_node)

    builder.add_edge(START, "rag_worker")
    builder.add_conditional_edges("rag_worker", supervisor_fanout, ["generate_mongodb_query_node"])
    builder.add_edge("generate_mongodb_query_node", "execute_mongodb_query_node")
    builder.add_edge("execute_mongodb_query_node", "format_response_node")
    builder.add_edge("format_response_node", END)

    return builder.compile()
