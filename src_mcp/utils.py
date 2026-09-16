import os
import json
import time
import uuid
from pathlib import Path
from typing import Optional

from bson.json_util import dumps, loads
from dotenv import load_dotenv
from pymongo.errors import ExecutionTimeout, PyMongoError
import boto3
from botocore.config import Config
from langchain_aws import ChatBedrockConverse
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from src_mcp.config import ALLOWED_PIPELINE_STAGES, MAX_RESULTS, MODEL_ID, DB_HOST, DB_PORT, DB_NAME1, bedrock
from src_mcp.dependencies import logger
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - optional dependency; app should still boot
    ChatGoogleGenerativeAI = None
import motor.motor_asyncio
from botocore.exceptions import ClientError, BotoCoreError
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
from src_mcp.models import *
from src_mcp.config import LLM_TIMEOUT

UNIQUE_NAMES_DICT = {}
db = None


def validate_pipeline_stages(pipeline: list) -> Optional[str]:
    """Returns an error message if the pipeline contains a disallowed stage, else None."""
    if not isinstance(pipeline, list):
        return "Pipeline must be a JSON array of stage objects."
    for stage in pipeline:
        if not isinstance(stage, dict) or len(stage) != 1:
            return f"Malformed pipeline stage: {stage}"
        stage_name = next(iter(stage.keys()))
        if stage_name not in ALLOWED_PIPELINE_STAGES:
            return f"Disallowed pipeline stage: '{stage_name}'."
    return None

async def execute_mongo_query(mongo_query: list[dict], db: str, coll: str = "PnLCOB") -> list:
    """Executes a MongoDB aggregation pipeline string asynchronously."""

    if db is None:
        logger.error("Database handle is unavailable.")
        return "Database connection error."

    if not mongo_query:
        logger.error("No mongo query provided.")
        return "No mongo query is found."

    try:
        logger.info(f"Validating and executing async aggregation pipeline on {coll} collection.")

        pipeline = loads(dumps(mongo_query))
        
        validation_error = validate_pipeline_stages(pipeline)
        if validation_error:
            logger.error(f"Pipeline rejected: {validation_error}")
            return f"Pipeline Validation Error: {validation_error}"

        target_collection = db[coll]
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

async def get_unique_names_from_database():
    global UNIQUE_NAMES_DICT
    db = await startup_db_client()
    # logger.info("Getting Unique Proper Names...")
    
    fields = ('AssetClass', 'TradeName', 'AccountName', 'ThemeName', 'TraderName')
    
    for field in fields:
        # Fixed: Field reference inside $group needs a '$' prefix e.g., '$AssetClass'
        query_pipeline = [
            {'$group': {'_id': f'${field}'}},
            {'$project': {'_id': 0, field: '$_id'}}
        ]
        
        # Use await instead of asyncio.run() because we are already inside an async function
        results = await execute_mongo_query(mongo_query=query_pipeline, db=db)
        # Extract unique values from the query result
        for result in results:
            reversed_result = {value: key for key, value in result.items()}
            UNIQUE_NAMES_DICT.update(reversed_result)
    return UNIQUE_NAMES_DICT

def initialize_llm_client():
    if bedrock:
        try:
            session = boto3.Session()
            boto_config = Config(
                connect_timeout=10,   # seconds to establish initial socket connection
                read_timeout=LLM_TIMEOUT,     # seconds to wait for a response from Bedrock
                retries={"max_attempts": 3, "mode": "adaptive"},
            )

            bedrock_client = session.client("bedrock-runtime", config=boto_config)
            # MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
            llm = ChatBedrockConverse(
                model=MODEL_ID,
                client=bedrock_client,
                region_name=os.getenv("AWS_DEFAULT_REGION"),
                temperature=0.0,
                max_tokens=8192,
                disable_streaming=False,
                additional_model_request_fields={
                    "top_k":1,
                    "top_p":1,
                }
            )
            logger.info("Successfully initialized ChatBedrockConverse client.")
        except ClientError as e:
            llm = None
            if e.response['Error']['Code'] == 'ExpiredTokenException':
                logger.critical(f"The security token included in the request is expired", exc_info=True)
                final_text = f"The security token included in the request is expired"
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
        except Exception as e:
            logger.critical(f"Failed to initialize LLM client: {str(e)}", exc_info=True)
    else:
        try:
            llm = ChatNVIDIA(
                model="mistralai/mistral-nemotron",
                api_key=os.getenv("NVIDIA_API_KEY"),
                timeout=LLM_TIMEOUT
            )
            logger.info("Successfully initialized ChatNVIDIA client.")
            return llm
        except:
            llm = None
            logger.critical(f"Failed to initialize LLM client: {str(e)}", exc_info=True)

        # if ChatGoogleGenerativeAI is None:
        #     logger.warning("langchain-google-genai is not installed. LLM integration disabled.")
        #     return None
        # try:
        #     model_id = "gemini-3.5-flash"
        #     llm = ChatGoogleGenerativeAI(model=model_id, temperature=0.0, request_timeout=5000)
        #     logger.info("Successfully initialized ChatGoogleGenerativeAI client.")
        # except Exception as e:
        #     llm = None
        #     logger.critical(f"Failed to initialize LLM client: {str(e)}", exc_info=True)

    return llm


async def startup_db_client():
    global db
    """Verify Mongo connection asynchronously on application startup."""
    db = None
    client = None
    connection_string = f"mongodb://{DB_HOST}:{DB_PORT}/{DB_NAME1}?authSource=admin"

    try:
        logger.info("Initializing Async Motor MongoDB Client...")
        client = motor.motor_asyncio.AsyncIOMotorClient(connection_string, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME1]
        logger.info(f"Initialized Motor Client for database: '{DB_NAME1}'")
        return db
    except Exception as e:
        logger.critical(f"Database Connection Error: {str(e)}", exc_info=True)
        return None

def convert_messages_to_string(messages: list) -> str:
    formatted_lines = []

    for msg in messages:
        # 1. Handle LangChain Message objects (HumanMessage, AIMessage, SystemMessage)
        if isinstance(msg, BaseMessage):
            role = msg.type.capitalize()  # 'human' -> 'Human', 'ai' -> 'Ai'
            content = msg.content

        # 2. Handle Pydantic models (like request.messages from FastAPI)
        elif isinstance(msg, BaseModel):
            role = getattr(msg, "role", "User").capitalize()
            content = getattr(msg, "content", "")

        # 3. Handle standard Python dictionaries
        elif isinstance(msg, dict):
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")

        else:
            continue

        formatted_lines.append(f"{role}: {content}")

    return "\n\n".join(formatted_lines)