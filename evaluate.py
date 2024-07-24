import weave
import asyncio

from mixeval.models.gpt_4o import GPT_4o
from mixeval.metrics.metrics import FreeForm

weave.init("ayush-thakur/weave-mixeval")

dataset = weave.ref(
    "weave:///ayush-thakur/weave-mixeval/object/mixeval-hard_free-form.json:o5E0ga2MNzNeP1YwLkKwpexcsUg2FlFPQX0THWKfBYI"
).get()

print(len(dataset.rows))

model = GPT_4o()

scorers: list = [FreeForm()]

evaluation = weave.Evaluation(
    dataset=dataset.rows, scorers=scorers
)

if __name__ == "__main__":
    asyncio.run(evaluation.evaluate(model))
