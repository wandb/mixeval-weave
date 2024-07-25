import os
import time
import random
from httpx import Timeout

from mixeval.models.base_api import APIModelBase
from mixeval.utils.registry import register_model

import anthropic
from anthropic._exceptions import RateLimitError

import weave

client = anthropic.AsyncAnthropic(
    api_key=os.getenv('k_ant'),
    timeout=Timeout(timeout=20.0, connect=5.0)
)


@register_model("claude_3_5_sonnet")
class Claude_3_5_Sonnet(APIModelBase):
    FIX_INTERVAL_SECOND: int = 1
    model_name: str = 'claude-3-5-sonnet-20240620'

    @weave.op()
    async def _decode(self, inputs: list):
        completion = await client.messages.create(
            model=self.model_name,
            max_tokens=self.MAX_NEW_TOKENS,
            messages=inputs,
        )
        time.sleep(self.FIX_INTERVAL_SECOND)
        return completion.content[0].text

    @weave.op()
    async def decode(self, inputs: list):
        delay = 1
        for i in range(self.MAX_RETRY_NUM):
            try:
                response_content = await self._decode(inputs)
                return response_content
            except RateLimitError as e:
                exponential_base = 2
                delay *= exponential_base * (1 + random.random())
                print(
                    f"RateLimitError, retrying after {round(delay, 2)} seconds, {i+1}-th retry..."
                )
                print(e)
                time.sleep(delay)
                continue
            except Exception as e:
                print(f"Error in decode, retrying...")
                print(e)
                time.sleep(1)
                continue
        print(f"Failed after {self.MAX_RETRY_NUM} retries.")
        return "Error"
