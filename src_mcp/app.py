import os
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import boto3
import motor.motor_asyncio
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_aws import ChatBedrockConverse

from src_mcp.config import *
from src_mcp.dependencies import logger
from src_mcp.models import *
from src_mcp.orchestrator import build_orchestrator

load_dotenv(override=True)


def openwebui_status(description: str, done: bool = False):
    """
    Build an Open WebUI-compatible status event.
    """
    return {
        "id": f"status-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": ""
            },
            "finish_reason": None
        }],
        "status": {
            "description": description,
            "done": done
        }
    }

# -------------------------------------------------------------------
# 6. FastAPI Application Initialization & Route Handling
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing reusable graph resources")
    app.state.graph = await build_orchestrator()
    yield
    app.state.graph = None


app = FastAPI(title="MongoDB Dynamic Query Engine (Async OpenAI Endpoint)", lifespan=lifespan)

# NOTE: allow_credentials=True cannot be safely combined with a wildcard
# origin (browsers reject it). Restrict allow_origins to real values if
# credentials are ever needed; wildcard here assumes no credentialed requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# API Key Authentication Setup
# -------------------------------------------------------------------
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="PNL-CHAT-API-KEY", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

async def verify_api_key(
    api_key: Optional[str] = Depends(api_key_header),
    bearer_credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
):
    """Verifies incoming requests against either PNL-CHAT-API-KEY or Bearer Token."""
    token = api_key

    # Extract token from standard 'Authorization: Bearer <key>' if present
    if not token and bearer_credentials:
        token = bearer_credentials.credentials

    if not token or token != API_KEY:
        final_text = "Invalid or missing API Key"
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
    return token

# @app.on_event("startup")
# async def startup_db_client():
#     global db
#     """Verify Mongo connection asynchronously on application startup."""
#     connection_string = f"mongodb://{DB_HOST}:{DB_PORT}/{DB_NAME}?authSource=admin"

#     try:
#         logger.info("Initializing Async Motor MongoDB Client...")
#         client = motor.motor_asyncio.AsyncIOMotorClient(connection_string, serverSelectionTimeoutMS=5000)
#         db = client[DB_NAME]
#         logger.info(f"Initialized Motor Client for database: '{DB_NAME}'")
#     except Exception as e:
#         logger.critical(f"Database Connection Error: {str(e)}", exc_info=True)

#     if client:
#         try:
#             await client.admin.command("ping")
#             logger.info("Pinged MongoDB cluster asynchronously. Connection verified.")
#         except Exception as e:
#             logger.critical(f"Async startup ping failed: {str(e)}")

# -------------------------------------------------------------------
# 3. LLM Initialization
# -------------------------------------------------------------------
# agent = None
# @app.on_event("startup")
# def initialize_llm_client():
#     global llm, agent
#     try:
#         session = boto3.Session()
#         boto_config = Config(
#             connect_timeout=10,   # seconds to establish initial socket connection
#             read_timeout=180,     # seconds to wait for a response from Bedrock
#             retries={"max_attempts": 3, "mode": "adaptive"},
#         )

#         bedrock_client = session.client("bedrock-runtime", config=boto_config)
#         # MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
#         llm = ChatBedrockConverse(
#             model=MODEL_ID,
#             client=bedrock_client,
#             region_name=os.getenv("AWS_DEFAULT_REGION"),
#             temperature=0.0,
#             max_tokens=8192,
#             disable_streaming=False,
#             additional_model_request_fields={
#                 "top_k":1,
#                 "top_p":1,
#             }
#         )
#         logger.info("Successfully initialized ChatBedrockConverse client.")
#         agent = startup_agent(llm)
#     except ClientError as e:
#         if e.response['Error']['Code'] == 'ExpiredTokenException':
#             logger.critical(f"The security token included in the request is expired", exc_info=True)
#             final_text = f"The security token included in the request is expired"
#             return ChatCompletionResponse(
#                 id=f"chatcmpl-{uuid.uuid4()}",
#                 created=int(time.time()),
#                 model=MODEL_ID,
#                 choices=[
#                     Choice(
#                         index=0,
#                         message=ChoiceMessage(role="assistant", content=final_text),
#                         finish_reason="stop",
#                     )
#                 ],
#                 usage=UsageInfo(),
#             )
#     except Exception as e:
#         logger.critical(f"Failed to initialize LLM client: {str(e)}", exc_info=True)
#     # try:
#     #     MODEL_ID = "gemini-3.1-flash-lite"
#     #     llm = ChatGoogleGenerativeAI(model=MODEL_ID, temperature=0.0, request_timeout=5000)
#     #     logger.info("Successfully initialized ChatGoogleGenerativeAI client.")
#     #     agent = startup_agent(llm)
#     # except Exception as e:
#     #     logger.critical(f"Failed to initialize LLM client: {str(e)}", exc_info=True)
#     #     final_text = f"Failed to initialize LLM client: {str(e)}"
#     #     return ChatCompletionResponse(
#     #         id=f"chatcmpl-{uuid.uuid4()}",
#     #         created=int(time.time()),
#     #         model=MODEL_ID,
#     #         choices=[
#     #             Choice(
#     #                 index=0,
#     #                 message=ChoiceMessage(role="assistant", content=final_text),
#     #                 finish_reason="stop",
#     #             )
#     #         ],
#     #         usage=UsageInfo(),
#     #     )

# @app.on_event("startup")
# async def get_unique_names_from_database():
#     global UNIQUE_NAMES_DICT
#     UNIQUE_NAMES_DICT = {}
    
#     logger.info("Getting Unique Proper Names...")
    
#     fields = ('AssetClass', 'TradeName', 'AccountName', 'ThemeName', 'TraderName')
    
#     for field in fields:
#         # Fixed: Field reference inside $group needs a '$' prefix e.g., '$AssetClass'
#         query_pipeline = [
#             {'$group': {'_id': f'${field}'}},
#             {'$project': {'_id': 0, field: '$_id'}}
#         ]
        
#         # Use await instead of asyncio.run() because we are already inside an async function
#         results = await execute_mongo_query.coroutine(mongo_query=query_pipeline, get_graph=False)
#         # Extract unique values from the query result
#         for result in results:
#             reversed_result = {value: key for key, value in result.items()}
#             UNIQUE_NAMES_DICT.update(reversed_result)

#     # with open('UNIQUE_NAMES_DICT.json','w', encoding="utf-8") as f:
#     #     json.dump(UNIQUE_NAMES_DICT, f)
#     # logger.info(f"All unique values loaded: {len(UNIQUE_NAMES_DICT)=}")


# -------------------------------------------------------------------
# Open WebUI Version Check Stubs
# -------------------------------------------------------------------
@app.get("/api/version")
@app.get("/v1/api/version")
@app.get("/v1/version")
async def version_stub():
    """Stub endpoint to prevent Open WebUI 404 health check errors."""
    return {"version": "0.1.0"}
    
@app.get("/v1/models")
async def list_models():
    """Endpoint required by Open WebUI to populate the model selection dropdown."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,  # Returns 'anthropic.claude-3-5-sonnet-20240620-v1:0'
                "name": "BLOTTER IQ 3.5 SONNET",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "bedrock",
            }
        ],
    }

@app.get("/api/tags")
@app.get("/v1/api/tags")
async def ollama_tags_stub():
    """Stub endpoint to prevent Open WebUI 404 errors when polling Ollama endpoints."""
    return {
        "models": [
            {
                "name": MODEL_ID,
                "model": MODEL_ID,
                "modified_at": "2026-09-02T00:00:00Z",
                "size": 0,
                "digest": "pnl-engine",
                "details": {"format": "gguf", "family": "claude"},
            }
        ]
    }

@app.get("/api/ps")
@app.get("/v1/api/ps")
async def ollama_ps_stub():
    """Stub endpoint for running process checks from Open WebUI."""
    return {"models": []}


@app.post("/api/chat", response_model=ChatCompletionResponse, dependencies=[Depends(verify_api_key)])
@app.post("/v1/api/chat", response_model=ChatCompletionResponse, dependencies=[Depends(verify_api_key)])
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(verify_api_key)])
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """OpenAI API v1 compatible POST route for completions (Fully Async)."""

    if not request.messages:
        logger.warning("Empty messages array provided in request payload.")
        final_text = "The request parameter 'messages' must contain at least one item."
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model or MODEL_ID,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=final_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(),
        )

    # Intercept Open WebUI Title Generation prompts
    if "Generate a concise title summarizing the chat history" in request.messages[-1].content:
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model or MODEL_ID,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant", 
                        content='{"title": "PnL Dashboard Query"}'
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(),
        )

    user_prompt = ""
    formatted_messages = []
    for msg in request.messages[-CONTEXT_MESSAGES_TO_CONSIDER:]:
        if msg.role == "user":
            formatted_messages.append(HumanMessage(content=msg.content))
            user_prompt = msg.content
        elif msg.role == "assistant":
            formatted_messages.append(AIMessage(content=msg.content))
    if not user_prompt or not user_prompt.strip():
        final_text = "The last message's 'content' must not be empty."
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=final_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(),
        )
    if len(user_prompt) > MAX_PROMPT_LENGTH:
        final_text = f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters."
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=final_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(),
        )

    logger.info(f"Received completion request: '{user_prompt}'")
    start_time = time.time()

    old_messages = []
    for msg in request.messages[:-CONTEXT_MESSAGES_TO_CONSIDER]:
        if msg.role == "user":
            old_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            old_messages.append(AIMessage(content=msg.content))

    # # 2. Get/create summary of old messages
    # summary = await get_conversation_summary(old_messages)

    # # 3. Build what Claude actually receives
    # formatted_messages = [
    #     {
    #         "role": "user",
    #         "content": f"""Here is a summary of the earlier conversation: {summary} """
    #     },
    #     *formatted_messages
    # ]
    
    # Honor the client's requested response mode.
    is_stream = request.stream is True
    graph = getattr(http_request.app.state, "graph", None)
    if graph is None:
        graph = await build_orchestrator()
        http_request.app.state.graph = graph

    try:
      if is_stream:
        async def event_generator():
            cmpl_id = f"chatcmpl-{uuid.uuid4()}"
            created_time = int(time.time())
            final_response = ""

            try:
                def status_chunk(description: str, done: bool = False) -> str:
                    return f"data: {json.dumps({
                        'id': cmpl_id,
                        'object': 'chat.completion.chunk',
                        'created': created_time,
                        'model': request.model or MODEL_ID,
                        'choices': [{
                            'index': 0,
                            'delta': {'role': 'assistant', 'content': f'{description}\n'},
                            'finish_reason': None
                        }],
                        'status': {'description': description, 'done': done}
                    })}\n\n"

                yield status_chunk("Analyzing your request...")

                initial_state = make_initial_agent_state(formatted_messages)

                async for update in graph.astream(initial_state, stream_mode="updates"):
                    for node_name, payload in update.items():
                        if isinstance(payload, dict):
                            final_response = payload.get("final_response", final_response)

                        if node_name == "rag_worker":
                            status = "📚 Relevant data sources found..."
                        elif node_name == "generate_mongodb_query_node":
                            status = "🔧 Generated MongoDB query..."
                        elif node_name == "execute_mongodb_query_node":
                            status = "⚡ Executed MongoDB query..."
                        elif node_name == "format_response_node":
                            status = "📊 Prepared results..."
                        else:
                            status = f"🔄 Updating {node_name}..."

                        yield status_chunk(status)

                final_response = final_response or "No matching records found in the database for the given criteria."
                yield f"data: {json.dumps({
                    'id': cmpl_id,
                    'object': 'chat.completion.chunk',
                    'created': created_time,
                    'model': request.model or MODEL_ID,
                    'choices': [{
                        'index': 0,
                        'delta': {'content': final_response},
                        'finish_reason': 'stop'
                    }],
                    'status': {'description': 'Final answer ready', 'done': True}
                })}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.error(f"Streaming error: {exc}", exc_info=True)
                yield f"data: {json.dumps({
                    'id': cmpl_id,
                    'object': 'chat.completion.chunk',
                    'created': created_time,
                    'model': request.model or MODEL_ID,
                    'choices': [{
                        'index': 0,
                        'delta': {'content': 'I could not process that request right now.'},
                        'finish_reason': 'stop'
                    }],
                    'status': {'description': 'Request failed', 'done': True}
                })}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
      else:
        logger.info(f"{formatted_messages=}")
        initial_state = make_initial_agent_state(formatted_messages)
        final_state = await graph.ainvoke(initial_state)
        # logger.info(f"final_text={final_state}")
        final_text = final_state.get("final_response") or "No matching records found in the database for the given criteria."
        logger.info(f"final_text={final_text}")
        # for msg in reversed(final_state["messages"][-CONTEXT_MESSAGES_TO_CONSIDER:]):
        #     # Capture Assistant Text
        #     if isinstance(msg, AIMessage) and not llm_analysis:
        #         if isinstance(msg.content, list):
        #             llm_analysis = "\n".join(
        #                 b["text"].strip()
        #                 for b in msg.content
        #                 if isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip()
        #             ) or None

        #         elif isinstance(msg.content, str):
        #             llm_analysis = msg.content
        # final_text = f"{final_text}\n\n{llm_analysis}"
        # logger.info(f"{final_text=}")
        end_time = time.time()
        logger.info(f"Request processed in {end_time - start_time:.2f} seconds.")
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=final_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(),
        )
    except (ClientError, BotoCoreError) as e:
        # Extract error code dynamically from response or class name
        error_code = ""
        if isinstance(e, ClientError):
            error_code = e.response.get("Error", {}).get("Code", "")
        else:
            error_code = e.__class__.__name__

        error_message = str(e)
        logger.error(f"Bedrock Error [{error_code}]: {error_message}", exc_info=True)

        # Check error code or class/string representations for Bedrock 503s
        if error_code in ("ServiceUnavailableException", "ServiceUnavailable") or "ServiceUnavailableException" in error_message:
            final_text = "The AI model service is currently unavailable. Please try again shortly."
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        message=ChoiceMessage(role="assistant", content=final_text),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(),
            )
        elif error_code in ("ThrottlingException", "TooManyRequestsException"):
            final_text = "Request rate limit exceeded. Please retry in a moment."
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        message=ChoiceMessage(role="assistant", content=final_text),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(),
            )
        else:
            final_text = f"AWS Bedrock Error: {error_message.split('URL')[0]}"
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                model=request.model,
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
        # Catch LangGraph wrapped task errors containing Bedrock ServiceUnavailable
        err_str = str(e)
        if "ServiceUnavailableException" in err_str or "ServiceUnavailable" in err_str:
            logger.error(f"Captured Bedrock 503 via LangGraph wrapper: {err_str}", exc_info=True)
            final_text = "The AI model service is currently unavailable. Please try again shortly."
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        message=ChoiceMessage(role="assistant", content=final_text),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(),
            )
        logger.critical(f"Server Execution Error: {err_str}", exc_info=True)
        final_text = f"Internal Server Error: {err_str}"
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=final_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(),
        )
