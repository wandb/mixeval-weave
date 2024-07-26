from mixeval.models.base_api import APIModelBase
from mixeval.utils.registry import register_model


@register_model("llama_3p1_70b_instruct")
class Llama3p170bInstruct(APIModelBase):
    model_name: str = "accounts/fireworks/models/llama-v3p1-70b-instruct"


if __name__ == "__main__":
    import weave
    import asyncio

    weave.init("ayush-thakur/weave-mixeval")

    dataset_row = weave.ref(
        "weave:///ayush-thakur/weave-mixeval/object/mixeval-hard_free-form.json:o5E0ga2MNzNeP1YwLkKwpexcsUg2FlFPQX0THWKfBYI/attr/rows/id/BnBZpkEigMohZMYpAs33nvYza8rgYKOhZRbPiUxGguc"
    ).get()
    print(dataset_row)

    model = Llama3p170bInstruct()
    output = asyncio.run(model.predict(**dict(dataset_row)))
    print(output)
