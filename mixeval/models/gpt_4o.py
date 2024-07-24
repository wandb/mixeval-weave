from mixeval.models.base_api import APIModelBase
from mixeval.utils.registry import register_model


@register_model("gpt_4o")
class GPT_4o(APIModelBase):
    model_name: str = 'gpt-4o-2024-05-13'


if __name__ == "__main__":
    import weave
    import asyncio

    weave.init("ayush-thakur/weave-mixeval")

    dataset_row = weave.ref(
        "weave:///ayush-thakur/weave-mixeval/object/mixeval-hard_free-form.json:o5E0ga2MNzNeP1YwLkKwpexcsUg2FlFPQX0THWKfBYI/attr/rows/id/BnBZpkEigMohZMYpAs33nvYza8rgYKOhZRbPiUxGguc"
    ).get()
    print(dataset_row)

    gpt_4o = GPT_4o()
    output = asyncio.run(gpt_4o.predict(**dict(dataset_row)))
    print(output)
