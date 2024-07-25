from mixeval.models.base_api import APIModelBase
from mixeval.utils.registry import register_model


@register_model("gpt_4o_mini")
class GPT_4o_Mini(APIModelBase):
    model_name: str = 'gpt-4o-mini'
