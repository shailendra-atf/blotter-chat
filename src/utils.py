from typing import Optional
from config import ALLOWED_PIPELINE_STAGES
import pandas as pd
from config import MAX_RESULTS
from dependencies import logger

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
