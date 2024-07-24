import os
import json
from openai import OpenAI
from openai import AsyncOpenAI
from httpx import Timeout
from tqdm import tqdm
import random
import time
from typing import Callable, Dict

from dotenv import load_dotenv
load_dotenv()

import weave

from openai._exceptions import RateLimitError
from mixeval.prompts.evaluation_prompts import (
    construct_prompt_freeform,
    construct_prompt_multichoice
)

# client = OpenAI(
#     api_key=os.getenv('k_oai'),
#     timeout=Timeout(timeout=100.0, connect=20.0)
# )

client = AsyncOpenAI(
    api_key=os.getenv('k_oai'),
    timeout=Timeout(timeout=100.0, connect=20.0)
)

# client = OpenAI(
#     base_url = "https://api.fireworks.ai/inference/v1",
#     api_key=os.getenv('FIREWORKS_API_KEY'),
#     timeout=Timeout(timeout=100.0, connect=20.0)
# )

# client = AsyncOpenAI(
#     base_url = "https://api.fireworks.ai/inference/v1",
#     api_key=os.getenv('FIREWORKS_API_KEY'),
#     timeout=Timeout(timeout=100.0, connect=20.0)
# )


class APIModelBase(weave.Model):
    FIX_INTERVAL_SECOND: int = 0
    MAX_RETRY_NUM: int = 512
    MAX_NEW_TOKENS: int = 1536
    get_user_message: Callable[[str], Dict[str, str]] = lambda prompt: {
        "role": "user",
        "content": prompt,
    }
    get_model_message: Callable[[str], Dict[str, str]] = lambda response: {
        "role": "assistant",
        "content": response,
    }

    @weave.op()
    async def _decode(self, inputs: list):
        completion = await client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "text"},
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

    @weave.op()
    async def predict(
        self,
        problem_type: str,
        context: str | None,
        prompt: str,
        target: list,
        benchmark_name: str,
        options: list | None = None,
    ):
        input = dict(
            problem_type=problem_type,
            prompt=prompt,
            benchmark_name=benchmark_name,
            context=context,
            target=target,
        )
        if options:
            input.update({"options": options})

        if problem_type == "free-form":
            formated_input = construct_prompt_freeform(input)
        elif problem_type == "multiple-choice":
            formated_input = construct_prompt_multichoice(input)
        else:
            raise NotImplementedError

        annotation = await self.decode([self.get_user_message(formated_input)])

        if annotation == "Error":
            input["response"] = None
            return input
        input["response"] = annotation

        return input
