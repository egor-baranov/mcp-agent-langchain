from typing import List, Dict
from langchain_community.chat_models import QwenChat
from .langchain import LangChain, LangChainAgent, RequestParams
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class QwenLangChain(LangChain):
    """Alibaba Qwen implementation for LangChain workflows"""

    def __init__(
            self,
            agent: LangChainAgent,
            api_key: str,
            context_variables: Dict[str, str] = None,
            **kwargs
    ):
        llm = QwenChat(
            api_key=api_key,
            model="qwen-max",
            **kwargs
        )
        super().__init__(
            agent=agent,
            llm=llm,
            context_variables=context_variables
        )

    async def generate(self, messages: List[Dict], params: RequestParams = None) -> str:
        params = params or RequestParams(
            temperature=0.4,
            max_tokens=6000,
            model="qwen-turbo"
        )
        return await super().generate(messages, params)

    async def stream(self, messages: List[Dict], params: RequestParams = None):
        params = params or RequestParams(temperature=0.4)
        formatted = self._format_messages(messages)
        async for chunk in self.llm.astream(formatted):
            yield chunk.content