from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

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
    user_prompt: str = Field(
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