import os
import time
import uuid
import json
from langchain_core.tools import tool
from bson.json_util import dumps, loads
from pymongo.errors import ExecutionTimeout, PyMongoError
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from models import GenerateQueryInput, MongoQuery, ExecuteQueryInput, ChatCompletionResponse, Choice, ChoiceMessage, UsageInfo, MarkdownDataInput, GraphResults
from config import MODEL_ID, MAX_RESULTS
from utils import validate_pipeline_stages
from context import *
from dependencies import logger

# -------------------------------------------------------------------
# 4. LangChain Tools Definition (Async)
# -------------------------------------------------------------------
@tool("generate_query", args_schema=GenerateQueryInput)
async def generate_query(user_prompt: str, system_prompt=QUERY_SYSTEM_PROMPT) -> MongoQuery:
    """Generates a valid MongoDB aggregation pipeline query (JSON array format)."""
    from app import llm
    from app import UNIQUE_NAMES_DICT
    if not user_prompt:
        final_text = "user_prompt parameter cannot be empty."
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=MODEL_ID,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=final_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(),
        )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "user",
                "give me query to {user_prompt}"
            ),
        ]
    )

    structured_llm = llm.with_structured_output(MongoQuery)
    chain = prompt_template | structured_llm
    response: MongoQuery = await chain.ainvoke(
        {
            "QUERY_GUARDRAILS_CONTEXT": QUERY_GUARDRAILS_CONTEXT,
            "CONTEXT": CONTEXT,
            "user_prompt": user_prompt,
            "COLLECTION": COLLECTION,
            "UNIQUE_NAMES_DICT": UNIQUE_NAMES_DICT
        }
    )

    logger.info(f"Generated Pipeline Query: {response.query}")
    return response.query

@tool("execute_mongo_query", args_schema=ExecuteQueryInput)
async def execute_mongo_query(mongo_query: list[dict], get_graph: bool =False) -> list:
    """Executes a MongoDB aggregation pipeline string asynchronously."""
    from app import db
    if db is None:
        logger.error("Database handle is unavailable.")
        return "Database connection error."

    if not mongo_query:
        logger.error("No mongo query provided.")
        return "No mongo query is found."

    try:
        logger.info(f"Validating and executing async aggregation pipeline on {COLLECTION} collection.")

        pipeline = loads(dumps(mongo_query))
        
        validation_error = validate_pipeline_stages(pipeline)
        if validation_error:
            logger.error(f"Pipeline rejected: {validation_error}")
            return f"Pipeline Validation Error: {validation_error}"

        target_collection = db[COLLECTION]
        cursor = target_collection.aggregate(pipeline, maxTimeMS=120000)
        mongo_results = await cursor.to_list()

        result_count = len(mongo_results)
        if result_count > MAX_RESULTS:
            return (
                f"Query returned {result_count} results, which exceeds the maximum limit of {MAX_RESULTS}. "
                "There are too many results. Please refine your query by getting sumn over the group."
            )
        elif result_count == 0:
            return (
                "No matching PnL records were found for the specified criteria."
            )

        if not mongo_results:
            logger.info("Query returned no matching records.")
            return "No matching PnL records were found for the specified criteria."

        # if get_graph:
        #     mongo_results = format_results(mongo_results)
        # logger.info(f"{mongo_results}")
        return mongo_results

    except ExecutionTimeout as e:
        logger.error(f"MongoDB Execution Timeout (maxTimeMS expired): {str(e)}")
        return "Query Timeout: The requested aggregation took too long to execute. Please narrow down your date range or filter criteria."

    except PyMongoError as e:
        logger.error(f"MongoDB Driver Error: {str(e)}", exc_info=True)
        return f"Database Query Error: {str(e)}"

    except json.JSONDecodeError as e:
        logger.error(f"JSON Parsing Error: {str(e)}")
        return f"JSON Parsing Error: Invalid MongoDB query payload. Details: {str(e)}"

    except Exception as e:
        logger.error(f"Execution Error: {str(e)}", exc_info=True)
        return f"Query Execution Error: {str(e)}"

@tool("plot_graph", args_schema=MarkdownDataInput)
async def plot_graph(markdown_data: str) -> GraphResults:
    """This function plots a mermaid graph from the data provided."""
    from app import llm
    if not markdown_data:
        final_text = "data not found."
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=MODEL_ID,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=final_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(),
        )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=GRAPH_SYSTEM_PROMPT),  # Literal text string, bypasses f-string parsing
            HumanMessagePromptTemplate.from_template("give me mermaid chart for to data: {markdown_data}"),
        ]
    )

    structured_llm = llm.with_structured_output(GraphResults)
    chain = prompt_template | structured_llm
    response: GraphResults = await chain.ainvoke(
        {
            "QUERY_GAURDRAILS_CONTEXT": GRAPH_SYSTEM_PROMPT,
            "markdown_data": markdown_data,
        }
    )
    mermaid_code = response.mermaid_code

    logger.info(f"{mermaid_code}")
    return GraphResults(
        mermaid_code=mermaid_code
    )


def startup_agent(llm):  
    agent = create_agent(
        model=llm,
        tools=[generate_query, execute_mongo_query, plot_graph],
        system_prompt=AGENT_SYSTEM_PROMPT
    )
    return agent