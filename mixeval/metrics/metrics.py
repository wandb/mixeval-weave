import numpy as np

import weave
from weave.flow.scorer import Scorer

from mixeval.judge_models.judge_freeform import GPTJudgeFreeForm


class FreeForm(Scorer):
    llm_judge: GPTJudgeFreeForm = GPTJudgeFreeForm()

    @weave.op()
    async def score(self, model_output: dict) -> dict:
        judge_output = await self.llm_judge.predict(model_output)

        return {
            "score": judge_output["judge_score"],
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
