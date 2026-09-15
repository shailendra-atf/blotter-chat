DB_HOST="10.1.2.40"
DB_PORT=27017
DB_NAME="ValuationSummary04June"
COLLECTION="PnLCOB"

# DB_HOST="localhost"
# DB_PORT=27017
# DB_NAME="PnLSummary"
# COLLECTION="FinalisedPnLSummary"

MAX_PROMPT_LENGTH =4000
MAX_RESULTS = 5000
CONTEXT_MESSAGES_TO_CONSIDER = 5  # Limit to last N messages for context

MODEL_ID ="anthropic.claude-3-5-sonnet-20240620-v1:0"

ALLOWED_PIPELINE_STAGES = {
    "$match", "$project", "$group", "$sort", "$limit", "$skip",
    "$unwind", "$lookup", "$addFields", "$set", "$count",
    "$facet", "$bucket", "$bucketAuto", "$sortByCount", "$replaceRoot",
    "$replaceWith", "$unset", "$sample",
}
