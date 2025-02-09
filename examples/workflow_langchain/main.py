import asyncio
from typing import List
from mcp_agent.agents.agent import Agent
from mcp_agent.types import Tool, CallToolResult
from mcp_agent.workflows.langchain.langchain_gigachat import GigaChatLangChain
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class ExampleAgent(Agent):
    """Example agent for GigaChat integration demo"""

    def __init__(self):
        super().__init__(
            name="GigaChatAssistant",
            instruction="You are an AI assistant powered by GigaChat",
            server_names=[],
            functions=[self.create_search_tool()]
        )

    def create_search_tool(self) -> dict:
        return {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                }
            }
        }

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
        logger.info(f"Calling tool {name} with {arguments}")
        return CallToolResult(
            content=[{"type": "text", "text": f"Results for {name}({arguments})"}]
        )


async def main():
    # Initialize components
    agent = ExampleAgent()
    await agent.initialize()

    giga_llm = GigaChatLangChain(
        agent=agent,
        credentials="YOUR_GIGACHAT_CREDENTIALS"  # Replace with actual credentials
    )

    # Example conversation
    messages = [
        {"role": "user", "content": "What's the weather in Moscow today?"},
        {"role": "assistant", "content": "Should I use web_search?"},
        {"role": "tool_call", "content": "web_search(query='Moscow weather')"}
    ]

    response = await giga_llm.generate(messages)
    print("Assistant Response:", response)


if __name__ == "__main__":
    asyncio.run(main())