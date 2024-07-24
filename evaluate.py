import weave
import asyncio

dataset = weave.ref(
    "weave:///ayush-thakur/weave-mixeval/object/mixeval-hard_free-form.json:o5E0ga2MNzNeP1YwLkKwpexcsUg2FlFPQX0THWKfBYI"
).get()

print(len(dataset.rows))

model = None

scorers: list = []

evaluation = weave.Evaluation(
    dataset=dataset, scorers=[scorers]
)

if __name__ == "__main__":
    asyncio.run(evaluation.evaluate(model))
