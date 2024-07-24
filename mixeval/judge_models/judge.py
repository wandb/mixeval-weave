import os
from httpx import Timeout

from openai import OpenAI


client = OpenAI(
    api_key=os.getenv('MODEL_PARSER_API'),
    timeout=Timeout(timeout=60.0, connect=5.0)
)


class GPTJudge:
    judge_model: str
    FIX_INTERVAL_SECOND: int = 0
    MAX_RETRY_NUM: int = 99
    MAX_NEW_TOKENS: int = 999

