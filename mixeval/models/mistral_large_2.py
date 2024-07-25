import os
import time
import random
from httpx import Timeout

from mixeval.models.base_api import APIModelBase
from mixeval.utils.registry import register_model

from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage

import weave

# pip install mistralai

client = MistralAsyncClient(
    api_key=os.getenv('k_mis'),
    timeout=Timeout(timeout=120.0, connect=5.0)
)


@register_model("mistral-large-2407")
class Mistral_Large_2(APIModelBase):
    FIX_INTERVAL_SECOND: int = 1
    model_name: str = 'mistral-large-2407'

    @weave.op()
    async def _decode(self, inputs: list):
        inputs = [
            ChatMessage(role=message['role'], content=message['content']) for message in inputs
        ]

        completion = await client.chat(
            model=self.model_name,
            max_tokens=self.MAX_NEW_TOKENS,
            messages=inputs,
        )
        time.sleep(self.FIX_INTERVAL_SECOND)
        return completion.choices[0].message.content

    @weave.op()
    async def decode(self, inputs: list):
        delay = 1
        for i in range(self.MAX_RETRY_NUM):
            try:
                response_content = await self._decode(inputs)
                return response_content
            except Exception as e:
                if 'rate' in str(e).lower():
                    exponential_base = 2
                    delay *= exponential_base * (1 + random.random())
                    print(f"Rate limit error, retrying after {round(delay, 2)} seconds, {i+1}-th retry...")
                    print(e)
                    time.sleep(delay)
                    continue
                else:
                    print(f"Error in decode, retrying...")
                    print(e)
                    time.sleep(1)
                    continue
        print(f"Failed after {self.MAX_RETRY_NUM} retries.")
        return "Error"
