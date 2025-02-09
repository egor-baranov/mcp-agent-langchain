from typing import List, Dict
from langchain_community.chat_models import ChatOpenAI
from .langchain import LangChain, LangChainAgent, RequestParams
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class OpenAILangChain(LangChain):
    """OpenAI implementation for LangChain workflows"""

    def __init__(
            self,
            agent: LangChainAgent,
            api_key: str,
            context_variables: Dict[str, str] = None,
            **kwargs
    ):
        llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4-turbo",
            **kwargs
        )
        super().__init__(
            agent=agent,
            llm=llm,
            context_variables=context_variables
        )

    async def generate(self, messages: List[Dict], params: RequestParams = None) -> str:
        params = params or RequestParams(
            temperature=0.7,
            max_tokens=4096,
            model="gpt-4-turbo"
        )
        return await super().generate(messages, params)

    async def stream(self, messages: List[Dict], params: RequestParams = None):
        params = params or RequestParams(temperature=0.7)
        formatted = self._format_messages(messages)
        async for chunk in self.llm.astream(formatted):
            yield chunk.content