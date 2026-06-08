from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, List, Type, Union

from pydantic import BaseModel

from nemantix.common.logger import get_package_logger
from nemantix.core.custom_types import PathLike
from nemantix.llm.abstract_proxy import AbstractLLMProxy, LLMProxyException

if TYPE_CHECKING:
    from nemantix.core.tools import Toolset

logger = get_package_logger(__name__)


class LocalLLMProxy(AbstractLLMProxy):
    """An LLM proxy for local language models (llama.cpp backend)"""

    def __init__(self, weights_path: PathLike, context_size=16_384,
                 temperature=1.0, top_p=0.95, max_tokens=2048,
                 role_start_token='', role_stop_token='', **llama_cpp_kwargs):
        import outlines
        from llama_cpp import Llama

        self.role_start_token = role_start_token  # e.g., <start_of_turn>
        self.role_stop_token = role_stop_token  # e.e.g, <end_of_turn>
        self.model_name = Path(weights_path).name

        self._model = Llama(model_path=str(weights_path), n_ctx=int(context_size),
                            verbose=False, **llama_cpp_kwargs)
        self.llm = outlines.models.LlamaCpp(self._model)
        self.params = None

        self.temperature = float(temperature)
        self.top_p = float(top_p)

        assert self.temperature >= 0
        assert 0.0 < self.top_p <= 1.0
        logger.info(f'Context size: {context_size}; temperature: {temperature}; top_p: {top_p}')

    def get_name(self) -> str:
        return f'Local {self.model_name}'

    def messages_from(self, prompts_with_roles: list[dict[str, str] | tuple[str, str]]) -> list[dict]:
        raise NotImplementedError('Not supported.')

    def invoke(self, prompt: str | list, **kwargs: Any) -> dict:
        if isinstance(prompt, list):
            prompt = self._convert_messages_to_string(messages=prompt)

        if self.params is None:
            text = self.llm(prompt, temperature=self.temperature, top_p=self.top_p)
        else:
            text = self.llm(prompt, sampling_params=self.params)

        return {"text": text, "tool_calls": []}

    def invoke_grammar_based(self, prompt: Union[str, list], **kwargs: Any) -> dict:
        raise LLMProxyException('Not supported')

    def invoke_structured(self, prompt: str | list, schema: Type[BaseModel], **kwargs):
        raise LLMProxyException('Not supported')

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        raise LLMProxyException('Not supported')

    def supports_tool_use(self) -> bool:
        return False

    def bind_tools(self, toolset_class: Type['Toolset'], tool_names: List[str]) -> "AbstractLLMProxy":
        pass

    def unbind_tools(self) -> "AbstractLLMProxy":
        pass

    def _convert_messages_to_string(self, messages: list) -> str:
        prompt = ''
        for msg in messages:
            prompt += (f'{self.role_start_token}{msg['role']}\n{msg['content'][0]['text']}'
                       f'{self.role_stop_token}')

        return prompt + f'{self.role_start_token}model\n'
