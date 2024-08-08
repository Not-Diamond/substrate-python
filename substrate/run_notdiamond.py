import os
from typing import List
from typing_extensions import Literal

from .nodes import ComputeText
from .run_python import RunPython

ComputeModels = Literal[
    "Mistral7BInstruct",
    "Mixtral8x7BInstruct",
    "Llama3Instruct8B",
    "Llama3Instruct70B",
    "Llama3Instruct405B",
    "Firellava13B",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-5-sonnet-20240620",
]


class RouteNotDiamond(RunPython):
    def __init__(self, route_input: ComputeText, models: List[ComputeModels], *args, **kwargs):
        super().__init__(
            function=model_select,
            kwargs={
                'query': route_input.future.text,
                "api_key": os.getenv("NOTDIAMOND_API_KEY"),
                "models": models
            },
            pip_install=['notdiamond'],
            *args, **kwargs
        )


def model_select(query: str, api_key: str, models: List[ComputeModels]) -> ComputeModels:
    from notdiamond import NotDiamond

    _compute_model_to_llm_config = {
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "claude-3-5-sonnet-20240620": "anthropic/claude-3-5-sonnet-20240620",
    }
    _llm_config_str_to_compute_model = {
        v: k for k, v in _compute_model_to_llm_config.items()
    }

    llm_configs = [
        _compute_model_to_llm_config[model] for model in models
    ]

    notdiamond = NotDiamond(api_key=api_key or os.getenv("NOTDIAMOND_API_KEY"))
    session_id, provider = notdiamond.model_select(
        messages=[{"role": "user", "content": query}], model=llm_configs
    )
    print(f"Routing prompt for {session_id} to {provider}")
    return _llm_config_str_to_compute_model[str(provider)]