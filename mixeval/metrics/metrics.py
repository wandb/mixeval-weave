import numpy as np
import random

import weave
from weave.flow.scorer import Scorer

from mixeval.judge_models.judge_freeform import GPTJudgeFreeForm
from mixeval.judge_models.judge_multichoice import GPTJudgeMultiChoice


class FreeForm(Scorer):
    llm_judge: GPTJudgeFreeForm = GPTJudgeFreeForm()

    @weave.op()
    async def score(self, model_output: dict) -> dict:
        judge_output = await self.llm_judge.predict(model_output)

        return {
            "score": judge_output["judge_pred"],
            "judge_response": judge_output["judge_response"],
            "benchmark_name": judge_output["benchmark_name"],
        }

    @weave.op()
    def summarize(self, score_rows: weave.WeaveList) -> dict:
        """Aggregate all the scores that are calculated for each row by the scoring function.
        Args:
         - score_rows: a WeaveList object, nested dict of metrics and scores
        Returns:
         - nested dict with the same structure as the input"""

        overall_score = [
            x.get("score") for x in score_rows if x.get("score") is not None
        ]

        _benchmark_score_dict = {}
        for score_row in score_rows:
            benchmark_name = score_row.get("benchmark_name")
            if benchmark_name is not None:
                if benchmark_name not in _benchmark_score_dict:
                    _benchmark_score_dict[benchmark_name] = []
                _benchmark_score_dict[benchmark_name].append(score_row.get("score"))

        count_overall_true = list(overall_score).count(True)
        overall_score = np.mean(overall_score)

        benchmark_score_dict = {}
        for k, v in _benchmark_score_dict.items():
            benchmark_score_dict[k] = {}
            benchmark_score_dict[k]["true_count"] = list(v).count(True)
            benchmark_score_dict[k]["true_fraction"] = np.mean(v)

        result = {
            "overall": {
                "true_count": count_overall_true,
                "true_fraction": overall_score,
            }
        }
        result.update(benchmark_score_dict)

        return result


class MultiChoice(Scorer):
    llm_judge: GPTJudgeMultiChoice = GPTJudgeMultiChoice()

    @weave.op()
    async def score(self, model_output: dict) -> dict:
        judge_output = await self.llm_judge.predict(model_output)

        return {
            "option": judge_output["judge_pred"],
            "judge_response": judge_output["judge_response"],
            "benchmark_name": judge_output["benchmark_name"],
            "target": judge_output["target"],
            "options": judge_output["options"],
        }

    @weave.op()
    def summarize(self, score_rows: weave.WeaveList) -> dict:
        """Aggregate all the scores that are calculated for each row by the scoring function.
        Args:
         - score_rows: a WeaveList object, nested dict of metrics and scores
        Returns:
         - nested dict with the same structure as the input"""

        _score_dict_model = {}
        for score_row in score_rows:
            options = score_row.get("options")
            target = score_row.get("target")
            assert isinstance(target, list) and len(target) == 1, \
                    f"Invalid target: {target}"
            all_choices = [chr(ord("A") + i) for i in range(len(options))]
            model_choice = score_row.get("option")
            target_id = all_choices[target[0]]
            judge_score = 1 if eval_multi_choice(target_id, model_choice) else 0
            
            if 'overall' not in _score_dict_model:
                _score_dict_model['overall'] = []
            _score_dict_model['overall'].append(judge_score)
            
            benchmark_name = score_row.get("benchmark_name")
            if benchmark_name not in _score_dict_model:
                _score_dict_model[benchmark_name] = []
            _score_dict_model[benchmark_name].append(judge_score)

        result = {}
        for k, v in _score_dict_model.items():
            result[k] = {}
            result[k]["true_count"] = list(v).count(True)
            result[k]["true_fraction"] = np.mean(v)

        return result


class MixEvalScorer(Scorer):
    freeform_llm_judge: GPTJudgeFreeForm = GPTJudgeFreeForm()
    multichoice_llm_judge: GPTJudgeMultiChoice = GPTJudgeMultiChoice()

    @weave.op()
    async def score(self, model_output: dict) -> dict:
        if model_output["problem_type"] == "free-form":
            judge_output = await self.freeform_llm_judge.predict(model_output)
            pred = judge_output.get("judge_pred")
            return {
                "judge_pred": pred,
                "judge_response": judge_output["judge_response"],
                "benchmark_name": judge_output["benchmark_name"],
                "target": judge_output["target"],
                "options": [],
                "problem_type": judge_output["problem_type"],
            }
        elif model_output["problem_type"] == "multiple-choice":
            judge_output = await self.multichoice_llm_judge.predict(model_output)
            pred = judge_output.get("judge_pred")
            options = judge_output["options"]
            return {
                "judge_pred": pred,
                "judge_response": judge_output["judge_response"],
                "benchmark_name": judge_output["benchmark_name"],
                "target": judge_output["target"],
                "options": options,
                "problem_type": judge_output["problem_type"],
            }
        else:
            raise NotImplementedError

    @weave.op()
    def summarize(self, score_rows: weave.WeaveList) -> dict:
        """Aggregate all the scores that are calculated for each row by the scoring function.
        Args:
         - score_rows: a WeaveList object, nested dict of metrics and scores
        Returns:
         - nested dict with the same structure as the input"""

        _score_dict_model = {}
        for score_row in score_rows:
            if "overall" not in _score_dict_model:
                _score_dict_model["overall"] = []

            benchmark_name = score_row.get("benchmark_name")
            if benchmark_name is not None:
                if benchmark_name not in _score_dict_model:
                    _score_dict_model[benchmark_name] = []
    
            if score_row["problem_type"] == "free-form":
                score = score_row.get("judge_pred")
                _score_dict_model["overall"].append(score)
                _score_dict_model[benchmark_name].append(score)
            elif score_row["problem_type"] == "multiple-choice":
                score = score_row.get("judge_pred")
                _score_dict_model['overall'].append(score)
                _score_dict_model[benchmark_name].append(score)
            else:
                raise NotImplementedError

        result = {}
        for k, v in _score_dict_model.items():
            result[k] = {}
            result[k]["true_count"] = list(v).count(True)
            result[k]["true_fraction"] = np.mean(v)

        return result
