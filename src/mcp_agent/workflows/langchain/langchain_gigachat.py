from typing import List, Dict
from mcp_agent.workflows.swarm.swarm import Swarm
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_gigachat import GigaChatAugmentedLLM
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class GigaChatLangChain(Swarm, GigaChatAugmentedLLM):
    """GigaChat implementation of LangChain workflow"""

    async def generate(self, message, request_params: RequestParams | None = None):
        params = self.get_request_params(
            request_params,
            default=RequestParams(
                model="GigaChat-Pro",
                maxTokens=8192,
                parallel_tool_calls=True,
            ),
        )
        iterations = 0
        response = None
        agent_name = str(self.aggregator.name) if self.aggregator else None

        while iterations < params.max_iterations and self.should_continue():
            response = await super().generate(
                message=message if iterations == 0 else "Continue processing request",
                request_params=params.copy(update={"max_iterations": 1})
            )

            logger.debug(f"GigaChat Agent: {agent_name}, response:", data=response)
            agent_name = self.aggregator.name if self.aggregator else None
            iterations += 1

        return response
