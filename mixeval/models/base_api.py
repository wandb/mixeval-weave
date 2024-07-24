import json
from tqdm import tqdm
import random
import time
from typing import Callable, Dict

import weave

from concurrent.futures import ThreadPoolExecutor
from openai._exceptions import RateLimitError


class APIModelBase(weave.Model):
    FIX_INTERVAL_SECOND: int = 0
    MAX_RETRY_NUM: int = 512
    MAX_NEW_TOKENS: int = 1536
    get_user_message: Callable[[str], Dict[str, str]] = lambda prompt: {"role": "user", "content": prompt}
    get_model_message: Callable[[str], Dict[str, str]] = lambda response: {"role": "assistant", "content": response}

    @weave.op()
    def _decode(self, inputs: list):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "text"},
            max_tokens=self.MAX_NEW_TOKENS,
            messages=inputs,
        )
        time.sleep(self.FIX_INTERVAL_SECOND)
        return completion.choices[0].message.content

    @weave.op()
    def decode(self, inputs: list):
        delay = 1
        for i in range(self.MAX_RETRY_NUM):
            try:
                response_content = self._decode(inputs)
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
    def annotate_p_open(self, task_dict: dict):
        current_turn_id = task_dict["current_turn_id"]
        messages = []
        # history
        for turn_id in range(current_turn_id):
            assert (
                task_dict["response"][turn_id] is not None
            ), "The response should not be None."
            messages.append(self.get_user_message(task_dict["turns"][turn_id]))
            messages.append(self.get_model_message(task_dict["response"][turn_id]))
        # current turn
        messages.append(self.get_user_message(task_dict["turns"][current_turn_id]))

        annotation = self.decode(messages)
        if "response" not in task_dict:
            task_dict["response"] = [None] * len(task_dict["turns"])
        if annotation == "Error":
            print(f"Error in decode, the entry {task_dict} will be retried later...")
            task_dict["response"][current_turn_id] = None
            return task_dict
        task_dict["response"][current_turn_id] = annotation
        return task_dict

    @weave.op()
    def annotate_p(self, task_dict: dict):
        return self.annotate_p_open(task_dict)

    def annotate_parallel(self, tasks):
        print(f"Generating response in parallel, in total {len(tasks)} threads.")
        results = []
        with ThreadPoolExecutor(len(tasks)) as executor:
            for entry in tqdm(executor.map(self.annotate_p, tasks), total=len(tasks)):
                results.append(entry)
        return results

    def get_openended_responses_turn(self, batch):
        task_dicts_valid = []
        task_dicts_remain = batch
        current_turn_id = batch[0]["current_turn_id"]

        while True:
            task_dicts = self.annotate_parallel(task_dicts_remain)
            task_dicts_remain = []
            for task_dict in task_dicts:
                if task_dict["response"][current_turn_id] is not None:
                    task_dicts_valid.append(task_dict)
                else:
                    task_dicts_remain.append(task_dict)
            if len(task_dicts_remain) == 0:
                break
            else:
                print(
                    f"Still {len(task_dicts_remain)} tasks remained to be predict. Retry..."
                )
        assert len(task_dicts_valid) == len(
            batch
        ), "The number of valid task_dicts should be the same as the input batch."
        return task_dicts_valid

    def get_openended_responses(self, batch):
        batch = [d["raw_inputs"] for d in batch]
        turn_num = len(batch[0]["turns"])
        for entry in batch:
            assert (
                len(entry["turns"]) == turn_num
            ), "All dialogues should have the same number of turns."

        for i in range(turn_num):
            for entry in batch:
                entry["current_turn_id"] = i
            batch = self.get_openended_responses_turn(batch)

    @weave.op()
    def predict(
        self,
        problem_type: str,
        context: str | None,
        prompt: str,
        target: list,
        benchmark_name: str,
    ):
        # preprocess the raw input
        
        

        return self.get_openended_responses(input)
