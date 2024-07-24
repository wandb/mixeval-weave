import os
import re
import ast
import time
import random
from httpx import Timeout

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI
from openai._exceptions import RateLimitError, BadRequestError

import weave

from mixeval.prompts.judge_prompts import gpt_judge_for_closeended_freeform

client = AsyncOpenAI(
    api_key=os.getenv("MODEL_PARSER_API"), timeout=Timeout(timeout=60.0, connect=5.0)
)

