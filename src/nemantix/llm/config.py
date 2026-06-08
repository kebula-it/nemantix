
from pathlib import Path

from nemantix.common.logger import get_package_logger
from nemantix.llm.abstract_proxy import AbstractLLMProxy
from nemantix.llm.credentials import Credentials
from nemantix.llm.factory import LLMProxyFactory

logger = get_package_logger(__name__)


class LLMProxyConfig:
    """Configure which LLM proxies to use"""

    def __init__(self, internal: dict | AbstractLLMProxy | None = None,
                 external: dict | AbstractLLMProxy | None = None,
                 summary: dict | AbstractLLMProxy | None = None,
                 knowledge_base: dict | AbstractLLMProxy | None = None,
                 coding: dict | AbstractLLMProxy | None = None,
                 default_vendor='openai', default_model='gpt-5-mini',
                 credentials_path: str | Path | None = None, **default_kwargs):
        if credentials_path is None:
            credentials_path = '.'

        credential_manager = Credentials.load_from_file(file_path=str(credentials_path))
        AbstractLLMProxy.set_credentials_manager(credential_manager)

        self.default_spec = dict(vendor=str(default_vendor), model=str(default_model),
                                 **default_kwargs)

        self._specs = dict(internal=internal or self.default_spec,
                           external=external or self.default_spec,
                           summary=summary or self.default_spec,
                           coding=coding or self.default_spec,
                           knowledge_base=knowledge_base or self.default_spec,
                           default=self.default_spec)
        self._proxies: dict[str, AbstractLLMProxy] = dict()

    def get(self, proxy: str) -> AbstractLLMProxy:
        proxy = str(proxy).strip().lower()

        if proxy not in self._proxies:
            if proxy not in self._specs:
                logger.warning(f'Key "{proxy}" not found in {[k for k in self._specs.keys()]}. '
                               f'Using default proxy.')

            spec_or_proxy = self._specs.get(proxy, self._proxies.get('default',
                                                                     self.default_spec))
            if isinstance(spec_or_proxy, AbstractLLMProxy):
                self._proxies[proxy] = spec_or_proxy

                if spec_or_proxy == self._proxies['default']:
                    logger.info(f'Using default LLM proxy for key "{proxy}".')
            else:
                assert isinstance(spec_or_proxy, dict)
                spec = spec_or_proxy
                llm_proxy = LLMProxyFactory.create_llm_proxy(vendor=spec['vendor'],
                                                             model_name=spec['model'],
                                                             **spec.get('kwargs', {}))
                if spec == self.default_spec:
                    logger.info(f'Using default LLM proxy for key "{proxy}".')

                self._proxies[proxy] = llm_proxy

        return self._proxies[proxy]
