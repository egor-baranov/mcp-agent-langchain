from typing import Callable, Dict, List, Optional, TypeVar, Generic, TYPE_CHECKING
from collections import defaultdict
from pydantic import AnyUrl, BaseModel, ConfigDict
from langchain_core.language_models import BaseChatModel

from mcp.types import (
    CallToolRequest,
    EmbeddedResource,
    CallToolResult,
    TextContent,
    TextResourceContents,
    Tool,
)
from mcp_agent.agents.agent import Agent
from mcp_agent.human_input.types import HumanInputCallback
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM, RequestParams
from mcp_agent.logging.logger import get_logger

if TYPE_CHECKING:
    from mcp_agent.context import Context

logger = get_logger(__name__)
MessageParamT = TypeVar("MessageParamT")
MessageT = TypeVar("MessageT")


class LangChainAgentResource(EmbeddedResource):
    """Resource containing LangChain agent reference"""
    agent: Optional["Agent"] = None
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class LangChainFunctionResultResource(EmbeddedResource):
    """Resource containing LangChain function execution result"""
    result: "LangChainFunctionResult"
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


def create_langchain_agent_resource(agent: "Agent") -> LangChainAgentResource:
    return LangChainAgentResource(
        type="langchain_resource",
        agent=agent,
        resource=TextResourceContents(
            text=f"Switching to Agent '{agent.name}'",
            uri=AnyUrl("http://langchain.url"),
        ),
    )


def create_langchain_function_result_resource(
        result: "LangChainFunctionResult"
) -> LangChainFunctionResultResource:
    return LangChainFunctionResultResource(
        type="langchain_result",
        result=result,
        resource=TextResourceContents(
            text=result.value or result.agent.name or "LangChainResult",
            uri=AnyUrl("http://langchain.url"),
        ),
    )


class LangChainAgent(Agent):
    """LangChain-enabled agent with full tool handling capabilities"""

    def __init__(
            self,
            name: str,
            instruction: str | Callable[[Dict], str] = "LangChain Agent",
            server_names: list[str] = None,
            functions: List["LangChainFunctionCallable"] = None,
            parallel_tool_calls: bool = True,
            human_input_callback: HumanInputCallback = None,
            context: Optional["Context"] = None,
            **kwargs,
    ):
        super().__init__(
            name=name,
            instruction=instruction,
            server_names=server_names,
            functions=functions,
            connection_persistence=False,
            human_input_callback=human_input_callback,
            context=context,
            **kwargs,
        )
        self.parallel_tool_calls = parallel_tool_calls

    async def call_tool(self, name: str, arguments: dict | None = None) -> CallToolResult:
        if not self.initialized:
            await self.initialize()

        if name in self._function_tool_map:
            tool = self._function_tool_map[name]
            result = await tool.run(arguments)

            logger.debug(f"LangChain tool {name} result:", data=result)

            if isinstance(result, Agent):
                return CallToolResult(content=[create_langchain_agent_resource(result)])
            elif isinstance(result, LangChainFunctionResult):
                return CallToolResult(
                    content=[create_langchain_function_result_resource(result)]
                )
            elif isinstance(result, (str, dict)):
                return CallToolResult(content=[TextContent(type="text", text=str(result))])

            logger.warning(f"Unhandled result type: {type(result)}")
            return CallToolResult(content=[TextContent(type="text", text=str(result))])

        return await super().call_tool(name, arguments)


class LangChainFunctionResult(BaseModel):
    """LangChain function execution result container"""
    value: str = ""
    agent: Agent | None = None
    context_variables: dict = {}
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


LangChainFunctionReturnType = str | Agent | dict | LangChainFunctionResult
LangChainFunctionCallable = Callable[[], LangChainFunctionReturnType]


class LangChain(AugmentedLLM[MessageParamT, MessageT], Generic[MessageParamT, MessageT]):
    """LangChain workflow orchestrator with full agent management"""

    def __init__(
            self,
            agent: LangChainAgent,
            llm: BaseChatModel,
            context_variables: Dict[str, str] = None
    ):
        super().__init__(agent=agent)
        self.llm = llm
        self.context_variables = defaultdict(str, context_variables or {})
        self.instruction = (
            agent.instruction(self.context_variables)
            if isinstance(agent.instruction, Callable)
            else agent.instruction
        )
        logger.debug(
            f"LangChain workflow initialized with {agent.name}",
            data={"context": self.context_variables, "instruction": self.instruction},
        )

    async def generate(self, messages: List[Dict], params: RequestParams) -> str:
        formatted_messages = self._format_messages(messages)
        response = await self.llm.agenerate(messages=[formatted_messages])
        return response.generations[0].message.content

    def _format_messages(self, messages: List[Dict]) -> List[Dict]:
        return [{
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        } for msg in messages]

    async def get_tool(self, tool_name: str) -> Tool | None:
        result = await self.aggregator.list_tools()
        return next((t for t in result.tools if t.name == tool_name), None)

    async def pre_tool_call(
            self, tool_call_id: str | None, request: CallToolRequest
    ) -> CallToolRequest | bool:
        if not self.aggregator:
            return False

        tool = await self.get_tool(request.params.name)
        if not tool:
            logger.warning(f"Tool {request.params.name} not found")
            return request

        if "context_variables" in tool.inputSchema:
            request.params.arguments["context_variables"] = self.context_variables
            logger.debug("Injected context variables into tool call")

        return request

    async def post_tool_call(
            self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult
    ) -> CallToolResult:
        processed = []
        for content in result.content:
            if isinstance(content, LangChainAgentResource):
                await self.set_agent(content.agent)
                processed.append(TextContent(type="text", text=content.resource.text))
            elif isinstance(content, LangChainFunctionResultResource):
                self.context_variables.update(content.result.context_variables)
                if content.result.agent:
                    await self.set_agent(content.result.agent)
                processed.append(TextContent(type="text", text=content.resource.text))
            else:
                processed.append(content)
        result.content = processed
        return result

    async def set_agent(self, agent: LangChainAgent):
        logger.info(f"Agent switch: {self.aggregator.name if self.aggregator else None} -> {agent.name}")
        if self.aggregator:
            await self.aggregator.shutdown()

        self.aggregator = agent
        if not agent or isinstance(agent, DoneLangChainAgent):
            self.instruction = None
            return

        await agent.initialize()
        self.instruction = (
            agent.instruction(self.context_variables)
            if callable(agent.instruction)
            else agent.instruction
        )

    def should_continue(self) -> bool:
        return bool(self.aggregator and not isinstance(self.aggregator, DoneLangChainAgent))


class DoneLangChainAgent(LangChainAgent):
    """Terminal agent for LangChain workflows"""

    def __init__(self):
        super().__init__(name="__langchain_done__", instruction="Workflow complete")

    async def call_tool(self, _: str, __: dict | None = None) -> CallToolResult:
        return CallToolResult(content=[TextContent(type="text", text="Workflow completed")])


async def create_langchain_transfer_tool(
        agent: "Agent", agent_function: Callable[[], None]
) -> Tool:
    return Tool(
        name="transfer_to_agent",
        description="Transfer control to another agent",
        agent_resource=create_langchain_agent_resource(agent),
        agent_function=agent_function,
    )


async def create_langchain_function_tool(
        agent_function: "LangChainFunctionCallable"
) -> Tool:
    return Tool(
        name="langchain_function",
        description="LangChain function tool",
        agent_resource=None,
        agent_function=agent_function,
    )