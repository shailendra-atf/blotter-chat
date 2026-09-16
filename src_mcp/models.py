from typing import List, Optional, Dict, Any, Annotated, TypedDict
from operator import add
from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    retrieved_docs: List[Dict[str, Any]]
    mongodb_queries: Annotated[List[Dict[str, Any]], add]
    query_results: Annotated[List[Dict[str, Any]], add]
    final_response: str


def make_initial_agent_state(messages: List[BaseMessage]) -> AgentState:
    return {
        "messages": messages,
        "retrieved_docs": [],
        "mongodb_queries": [],
        "query_results": [],
        "final_response": "",
    }


class TaskState(TypedDict):
    doc: Dict[str, Any]
    user_intent: str

class execute_read_queryOutput(BaseModel):
    # Change from: result: str
    result: Dict[str, Any]  # Or dict / Any
    
# class MasterState(TypedDict):
#     question: str
#     context: Annotated[List[str], add]       # Safely accumulates parallel db text outputs
#     schema_registry: Annotated[List[str], add] # Accumulates structural metadata discovered via RAG
#     loop_count: int
#     next_action: str                         # Determines graph path routing
#     tasks_to_run: List[dict]                 # Payload array for parallel workers

# class WorkerState(TypedDict):
#     task_type: str                           # 'DISCOVER_SCHEMA' or 'EXECUTE_QUERY'
#     payload: dict                            # Specific query strings or execution arguments

# ==========================================
# 1. EXPANDED PYDANTIC ROUTER SCHEMA
# ==========================================
# class RouterDecision(BaseModel):
#     decision: str = Field(
#         description=(
#             "Choose 'DISCOVER_SCHEMA' if you lack table signatures. "
#             "Choose 'EXECUTE_QUERY' if you have the schemas."
#             # "Choose 'EXECUTE_QUERY' if you have valid code/filters ready to execute. "
#             "Choose 'FINAL_ANSWER' if data context is complete."
#         )
#     )
#     reasoning: str = Field(description="Strategic logic breakdown for this specific graph hop.")

#     # Payload A: Plain-text lookups for RAG schema discovery
#     plain_text_intents: List[str] = Field(
#         default=[],
#         description=(
#             "List of semantic topics to look up in the RAG index. "
#             "Use these when the exact table or collection names are not known yet."
#         )
#     )

#     # Payload B: Compilation parameters for the query generator
#     queries_to_generate: List[dict] = Field(
#         default=[],
#         description=(
#             "List of schemas to translate into code. Each dict MUST contain: "
#             "'db_type' (SQL/MongoDB), 'target_resource' (table name), "
#             "'schema_info' (columns/fields string retrieved from RAG), and 'user_intent'."
#         )
#     )

#     # Payload C: Executable code strings ready to fire against databases
#     queries_to_execute: List[dict] = Field(
#         default=[],
#         description=(
#             "List of compiled queries to execute. Each dict MUST contain: "
#             "'db_type' (SQL/MongoDB), 'target_resource', and 'query_string' (valid raw SQL/Mongo dict string)."
#         )
#     )

# class ReActStepDecision(BaseModel):
#     thought: str = Field(
#         description="Internal monologue reasoning about what information or schema definitions are missing."
#     )
#     action: str = Field(
#         description="Choose 'CALL_MAPPING_RAG' to find tables/fields, or 'CONSTRUCT_SUCCESSFUL_PLAN' if you have located all targets."
#     )
#     search_keyword: str = Field(
#         default="", 
#         description="The clean search string to forward to the RAG metadata tool if calling the tool."
#     )
#     blueprint_conclusion: str = Field(
#         default="", 
#         description="The structured final plan containing database mappings, resource names, and schemas to return to the supervisor."
#     )

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str]
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False

class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: UsageInfo

class GenerateQueryInput(BaseModel):
    messages: str = Field(
        description="REQUIRED. The exact user prompt or request describing what aggregation to generate. MUST NOT be empty."
    )

class MongoQuery(BaseModel):
    query: List[Dict]

class ExecuteQueryInput(BaseModel):
    mongo_query: List[Dict[str, Any]] = Field(description="The MongoDB query object")
    get_graph: bool = Field(
        default=False, 
        description="Whether to format results for graph generation."
    )


class MarkdownDataInput(BaseModel):
    markdown_data: str

class GraphResults(BaseModel):
    mermaid_code: str