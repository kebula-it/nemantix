from nemantix.common.logger import get_package_logger
from nemantix.llm.abstract_proxy import AbstractLLMProxy
from nemantix.llm.credentials import Credentials
from nemantix.llm.factory import LLMProxyFactory

logger = get_package_logger(__name__)


class LLMProxyConfig:
    """Configure which LLM proxies to use"""

    # for type hints
    internal: AbstractLLMProxy  # request/deliberate resolution, in/out extraction, ...
    external: AbstractLLMProxy  # builtin ask_llm and do llm
    summary: AbstractLLMProxy  # coding summaries
    knowledge_base: AbstractLLMProxy
    coding: AbstractLLMProxy
    default: AbstractLLMProxy

    def __init__(
        self,
        internal: dict | AbstractLLMProxy | None = None,
        external: dict | AbstractLLMProxy | None = None,
        summary: dict | AbstractLLMProxy | None = None,
        knowledge_base: dict | AbstractLLMProxy | None = None,
        coding: dict | AbstractLLMProxy | None = None,
        default_vendor="openai",
        default_model="gpt-5-mini",
        **default_kwargs,
    ):
        credential_manager = Credentials()
        AbstractLLMProxy.set_credentials_manager(credential_manager)

        self.default_spec = dict(
            vendor=str(default_vendor), model=str(default_model), **default_kwargs
        )

        self._specs = dict(
            internal=internal or self.default_spec,
            external=external or self.default_spec,
            summary=summary or self.default_spec,
            coding=coding or self.default_spec,
            knowledge_base=knowledge_base or self.default_spec,
            default=self.default_spec,
        )
        self._proxies: dict[str, AbstractLLMProxy] = dict()

    def __getattr__(self, proxy: str) -> AbstractLLMProxy:
        return self.get(proxy)

    def get(self, proxy: str) -> AbstractLLMProxy:
        """Retrieves the configured proxy: instantiates it if required"""
        proxy = str(proxy).strip().lower()

        if proxy not in self._proxies:
            if proxy not in self._specs:
                logger.warning(
                    f'Key "{proxy}" not found in {[k for k in self._specs.keys()]}. '
                    f"Using default proxy."
                )

            spec_or_proxy = self._specs.get(
                proxy, self._proxies.get("default", self.default_spec)
            )
            if isinstance(spec_or_proxy, AbstractLLMProxy):
                self._proxies[proxy] = spec_or_proxy

                if spec_or_proxy == self._proxies.get("default", None):
                    logger.info(f'Using default LLM proxy for key "{proxy}".')
            else:
                assert isinstance(spec_or_proxy, dict)
                spec = spec_or_proxy

                if spec == self.default_spec and proxy != "default":
                    logger.info(f'Using default LLM proxy for key "{proxy}".')
                    llm_proxy = self.get(proxy="default")
                else:
                    logger.info(f'Instantiating LLM proxy for key "{proxy}".')
                    llm_proxy = LLMProxyFactory.create_llm_proxy(
                        vendor=spec["vendor"],
                        model_name=spec["model"],
                        **spec.get("kwargs", {}),
                    )

                self._proxies[proxy] = llm_proxy

        return self._proxies[proxy]
