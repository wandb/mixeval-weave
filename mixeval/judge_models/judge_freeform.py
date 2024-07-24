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


@weave.op()
def get_score_from_judge(judge_response):
    """
    Get the score from the judge response.
    """
    one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
    one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

    match = re.search(one_score_pattern, judge_response)
    if not match:
        match = re.search(one_score_pattern_backup, judge_response)

    if match:
        rating = ast.literal_eval(match.groups()[0])
    else:
        rating = -1
    
    return float(rating)


class GPTJudgeFreeForm:
    judge_model: str = "gpt-3.5-turbo-0125"
    FIX_INTERVAL_SECOND: int = 0
    MAX_RETRY_NUM: int = 99
    MAX_NEW_TOKENS: int = 999

    @weave.op()
    def format_prompts(self, inputs):
        prompt, gold_ans, response = inputs
        gold_ans = "; ".join(
            [f"<answer {i+1}> {ans}" for i, ans in enumerate(gold_ans)]
        )
        formated = gpt_judge_for_closeended_freeform(prompt, gold_ans, response)
        return formated

    @weave.op()
    async def _GPT_decode(self, inputs):
        completion = await client.chat.completions.create(
            model=self.judge_model,
            response_format={"type": "text"},
            max_tokens=self.MAX_NEW_TOKENS,
            messages=self.format_prompts(inputs),
        )
        time.sleep(self.FIX_INTERVAL_SECOND)
        return completion

    @weave.op()
    async def GPT_decode(self, inputs):
        delay = 1
        blocked = 0
        for i in range(self.MAX_RETRY_NUM):
            try:
                completion = await self._GPT_decode(inputs)
                # try to parse the score
                score = get_score_from_judge(completion.choices[0].message.content)
                if score == -1:
                    continue
                else:
                    return completion, score
            except RateLimitError as e:
                exponential_base = 2
                delay *= exponential_base * (1 + random.random())
                print(
                    f"RateLimitError, retrying after {round(delay, 2)} seconds, {i+1}-th retry..."
                )
                print(e)
                time.sleep(delay)
                continue
            except BadRequestError as e:
                blocked += 1
                if blocked >= 10:
                    print("Blocked too many times, skipping...")
                    return "Blocked", -1
                print(f"Input is blocked, retrying...")
                print(e)
                time.sleep(1)
                continue
            except Exception as e:
                print(f"Error in GPT_decode, retrying...")
                print(e)
                time.sleep(1)
                continue

        print(f"Failed after {self.MAX_RETRY_NUM} retries.")
        return "Error", -1

    @weave.op()
    async def predict(
        self,
        inputs: dict,
    ):
        prompt = inputs["prompt"]
        target = inputs["target"]
        response = inputs["response"]

        if not isinstance(target, list):
            print(f"Invalid target: {target}")
            return None

        _input = (prompt, target, response)

        completion, score = await self.GPT_decode(_input)
        if completion == "Error":
            print(f"Error in GPT_decode, the entry {_input} will be retried later...")
            inputs["judge_response"] = None
            inputs["judge_score"] = score
            return inputs
        elif completion == "Blocked":
            print(f"{input}: \n\nBlocked, the entry treated as bad entry.")
            inputs["judge_response"] = "[[0.0]]"
            inputs["judge_score"] = score
            return inputs
        annotation = completion.choices[0].message.content
        inputs["judge_response"] = annotation
        inputs["judge_score"] = score

        return inputs


if __name__ == "__main__":
    import weave
    import asyncio

    from mixeval.models.gpt_4o import GPT_4o

    weave.init("ayush-thakur/weave-mixeval")

    dataset_row = weave.ref(
        "weave:///ayush-thakur/weave-mixeval/object/mixeval-hard_free-form.json:o5E0ga2MNzNeP1YwLkKwpexcsUg2FlFPQX0THWKfBYI/attr/rows/id/BnBZpkEigMohZMYpAs33nvYza8rgYKOhZRbPiUxGguc"
    ).get()
    print(dataset_row)

    gpt_4o = GPT_4o()
    output = asyncio.run(gpt_4o.predict(**dict(dataset_row)))
    print(output)

    model_judge = GPTJudgeFreeForm()
    judged_output = asyncio.run(model_judge.predict(output))
    print(judged_output)
