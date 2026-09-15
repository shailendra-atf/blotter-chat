import asyncio
import sqlite3
import json

from bson.json_util import dumps, loads
from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient
from pymongo.errors import ExecutionTimeout, PyMongoError

from config import LIMIT_RESULTS, DB_NAME1
# from context import CONTEXT, QUERY_GUARDRAILS_CONTEXT, QUERY_SYSTEM_PROMPT
from dependencies import logger
from models import MongoQuery
from rag_ingestion import initialize_chromadb
from utils import startup_db_client, llm

load_dotenv(override=True)

db = startup_db_client()

mcp_server = FastMCP("Hybrid-RAG-DB-Server")
# ==========================================
# TOOL 1: RAG ROUTER FOR TABLE MATCHING
# ==========================================
# @mcp_server.tool()
# def find_relevant_tables(user_intent: str) -> str:
#     """
#     Searches the metadata vector registry to discover the exact SQL tables 
#     or MongoDB collections relevant to the user request. Returns schemas.
#     """
#     # Query vector database for top 2 most matching table descriptions
#     results = chroma_collection.query(query_texts=[user_intent], n_results=1)#min(1, chroma_collection.count()))
#     if not results or not results["metadatas"][0]:
#         return "No matching database schemas located for this intent."
        
#     output = ["### Located Matching Database targets via RAG:"]
#     for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
#         output.append(f"- DB Type: {meta['db_type']} | DB: {meta['db']} | Resource: {meta['target']}")
#         output.append(f"  Schema Structure: {meta['schema']}")
#         output.append(f"  Description: {doc}\n")
        
#     return "\n".join(output)

# ==========================================
# TOOL 2: QUERY GENERATOR
# ==========================================
@mcp_server.tool()
async def generate_mongodb_query(
    system_prompt: str,
    db_type: str,
    target_resource: str,
    schema_info: str,
    user_intent: str
) -> str:
    """
    Translates a plain-text user intent into a valid, executable SQL query string
    or MongoDB dictionary filter based on the target resource and its schema.
    """
    logger.info(f"{user_intent=}")
    user_intent = user_intent

    db_type_upper = db_type.upper()

    if db_type_upper == "SQL":
        prompt = (
            f"You are a SQL expert. Write a clean, read-only SQLite SELECT query for the table '{target_resource}'.\n"
            f"Table Schema: {schema_info}\n"
            f"User Intent: {user_intent}\n"
            f"Return ONLY the raw SQL code block. No explanations, no markdown formatting."
        )
        if llm is None:
            return "LLM client not initialized."
        sql_response = await llm.ainvoke(prompt)
        return sql_response.content if hasattr(sql_response, "content") else str(sql_response)

    elif db_type_upper == "MONGODB":
        if llm is None:
            return "LLM client not initialized."

        logger.info(f"{system_prompt=}")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Intent: {user_intent}\nResource: {target_resource}\n")
        ])
        logger.info(f"{prompt_template=}")
        chain = prompt_template | llm
        logger.info(f"{prompt_template=}")
        try:
            logger.info({
                    "user_intent": user_intent,
                    "target_resource": target_resource,
                })
            response = await asyncio.wait_for(
                chain.ainvoke({
                    "user_intent": user_intent,
                    "target_resource": target_resource,
                }),
                timeout=120,
            )
            
        except asyncio.TimeoutError:
            logger.error("Timed out while generating the MongoDB query from the LLM.")
            return "LLM query generation timed out after 120 seconds."
        logger.info(f"{response=}")
        content = response.content if hasattr(response, "content") else response
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content)

    return f"Unsupported db_type: {db_type}. Expected 'SQL' or 'MongoDB'."

# ==========================================
# TOOL 3: DUAL ENGINE QUERY EXECUTOR
# ==========================================

@mcp_server.tool()
def execute_hybrid_query(db_type: str, target_resource: str, query_string: str) -> str:
    """
    Executes a read-only query against SQL or MongoDB.
    db_type must be either 'SQL' or 'MongoDB'.
    query_string must be a SQL SELECT statement or a valid stringified MongoDB dictionary filter.
    """
    logger.info(f"{query_string=}")
    # 1. SQL Path Execution
    if db_type.upper() == "SQL":
        if not query_string.strip().lower().startswith("select"):
            return "Error: Security block. Only SELECT operations allowed on SQL."
        try:
            conn = sqlite3.connect("company_records.db")
            cursor = conn.cursor()
            cursor.execute(query_string)
            rows = cursor.fetchall()
            conn.close()
            return f"SQL Results from {target_resource}:\n{str(rows)}"
        except Exception as e:
            return f"SQL Error: {str(e)}"
            
    # 2. MongoDB Path Execution
    elif db_type.upper() == "MONGODB":
        try:
            # Connect to local or cluster mongo instance
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
            db = client[DB_NAME1]
            coll = db[target_resource]
            
            # Parse standard or Extended JSON without evaluating arbitrary code.
            query_obj = loads(query_string) if query_string.strip() else {}
            if isinstance(query_obj, list):
                mongo_results = list(
                    coll.aggregate(
                        query_obj,
                        allowDiskUse=True
                    )
                )
            else:
                mongo_results = list(
                    coll.find(query_obj).limit(LIMIT_RESULTS)
                )
            
            client.close()
            return f"MongoDB Documents from {target_resource}:\n{str(mongo_results)}"
        except Exception as e:
            return f"MongoDB Error: {str(e)}"
            
    return "Error: Unknown db_type specified."

if __name__ == "__main__":
    mcp_server.run(transport="stdio")
