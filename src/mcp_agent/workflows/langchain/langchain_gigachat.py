from typing import List, Dict
from langchain_community.chat_models import GigaChat
from .langchain import LangChain, LangChainAgent, RequestParams
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class GigaChatLangChain(LangChain):
    """GigaChat implementation for LangChain workflows"""

    def __init__(
            self,
            agent: LangChainAgent,
            credentials: str,
            context_variables: Dict[str, str] = None,
            **kwargs
    ):
        llm = GigaChat(
            credentials=credentials,
            verify_ssl_certs=False,
            scope="GIGACHAT_API_CORP",
            **kwargs
        )
        super().__init__(
            agent=agent,
            llm=llm,
            context_variables=context_variables
        )

    async def generate(self, messages: List[Dict], params: RequestParams = None) -> str:
        params = params or RequestParams(
            temperature=0.5,
            max_tokens=8192,
            model="GigaChat-latest"
        )
        return await super().generate(messages, params)

    async def stream(self, messages: List[Dict], params: RequestParams = None):
        params = params or RequestParams(temperature=0.5)
        formatted = self._format_messages(messages)
        async for chunk in self.llm.astream(formatted):
            yield chunk.content