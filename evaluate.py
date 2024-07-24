import weave
import asyncio

from mixeval.models.gpt_4o import GPT_4o
from mixeval.metrics.metrics import MixEvalScorer

weave.init("ayush-thakur/weave-mixeval")

mixeval_hard_free_form = weave.ref(
    "weave:///ayush-thakur/weave-mixeval/object/mixeval-hard_free-form.json:o5E0ga2MNzNeP1YwLkKwpexcsUg2FlFPQX0THWKfBYI"
).get()
mixeval_hard_multiple_choice = weave.ref(
    "weave:///ayush-thakur/weave-mixeval/object/mixeval-hard_multiple-choice.json:1KFc1sZp8hnt6wcE0jt9E2FqGNQ1Y45XFgqctNF8CqQ"
).get()
dataset = list(mixeval_hard_free_form.rows)+list(mixeval_hard_multiple_choice.rows)
print(len(dataset))

model = GPT_4o()

scorers: list = [MixEvalScorer()]

evaluation = weave.Evaluation(
    dataset=dataset, scorers=scorers
)

if __name__ == "__main__":
    asyncio.run(evaluation.evaluate(model))
