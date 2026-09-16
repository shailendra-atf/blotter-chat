DB_HOST="10.1.2.40"
DB_PORT=27017
DB_NAME1="ValuationSummary04June"
COLLECTION1="PnLCOB"
COLLECTION2="TraderFundCharges"
COLLECTION3="TraderLimit"
COLLECTION4="FundUSDAum"

# DB_HOST="localhost"
# DB_PORT=27017
# DB_NAME="PnLSummary"
# COLLECTION="FinalisedPnLSummary"

SQL_DB_HOST="10.1.2.40"
SQL_DB_PORT=27017
SQL_DB_NAME="blotterlive11Sept"
SQL_Table1=""
SQL_Table2=""

MAX_PROMPT_LENGTH =4000
MAX_RESULTS = 5000
LIMIT_RESULTS = 1000
CONTEXT_MESSAGES_TO_CONSIDER = 5  # Limit to last N messages for context
# SIMILARITY_THRESHOLD=.65
SIMILARITY_THRESHOLD=.3

MODEL_ID ="anthropic.claude-3-5-sonnet-20240620-v1:0"

ALLOWED_PIPELINE_STAGES = {
    "$match", "$project", "$group", "$sort", "$limit", "$skip",
    "$unwind", "$lookup", "$addFields", "$set", "$count",
    "$facet", "$bucket", "$bucketAuto", "$sortByCount", "$replaceRoot",
    "$replaceWith", "$unset", "$sample",
}

# Configure the endpoint for your HTTP SSE MCP Server
MCP_SERVER_SSE_URL = "http://127.0.0.1:8090/sse"

LLM_TIMEOUT=180
bedrock = True