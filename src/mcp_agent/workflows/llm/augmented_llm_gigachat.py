import json
from typing import List, Type

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from mcp.types import (
    CallToolRequestParams,
    CallToolRequest,
    EmbeddedResource,
    ImageContent,
    ModelPreferences,
    TextContent,
)
from mcp_agent.logging.logger import get_logger
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    ModelT,
    MCPMessageParam,
    MCPMessageResult,
    ProviderToMCPConverter,
    RequestParams,
)

logger = get_logger(__name__)


class GigaChatAugmentedLLM(AugmentedLLM[Messages, Chat]):
    """GigaChat implementation of AugmentedLLM"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, type_converter=MCPGigaChatTypeConverter, **kwargs)
        self.provider = "GigaChat"

        self.model_preferences = self.model_preferences or ModelPreferences(
            costPriority=0.3,
            speedPriority=0.4,
            intelligencePriority=0.3,
        )
        self.default_request_params = self.default_request_params or RequestParams(
            model="GigaChat-Plus",
            modelPreferences=self.model_preferences,
            maxTokens=8192,
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=10,
            use_history=True,
        )

    async def generate(self, message, request_params: RequestParams | None = None):
        config = self.context.config
        gigachat_client = GigaChat(
            credentials=config.gigachat.credentials,
            verify_ssl_certs=config.gigachat.verify_ssl,
            scope=config.gigachat.scope
        )

        messages: List[Messages] = []
        params = self.get_request_params(request_params)

        if params.systemPrompt:
            messages.append(Messages(
                role=MessagesRole.SYSTEM,
                content=params.systemPrompt
            ))

        if params.use_history:
            messages.extend(self.history.get())

        if isinstance(message, str):
            messages.append(Messages(
                role=MessagesRole.USER,
                content=message
            ))
        elif isinstance(message, list):
            messages.extend(message)
        else:
            messages.append(message)

        responses: List[Chat] = []
        model = await self.select_model(params)

        for i in range(params.max_iterations):
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": params.maxTokens,
                "temperature": params.temperature,
            }

            logger.debug(
                f"Iteration {i}: Calling GigaChat with messages:",
                data=messages,
            )

            try:
                response = await gigachat_client.achat(**payload)
                responses.append(response)

                if response.choices:
                    message = response.choices[0].message
                    messages.append(message)

                    if message.content:
                        self.handle_content(messages, message)

                    if hasattr(message, 'tool_calls'):
                        await self.handle_tool_calls(message.tool_calls, messages)

                if self.should_stop(response):
                    break

            except Exception as e:
                logger.error(f"GigaChat API error: {str(e)}")
                break

        if params.use_history:
            self.history.set(messages)

        return responses

    async def generate_str(self, message, request_params: RequestParams | None = None):
        responses = await self.generate(message, request_params)
        return "\n".join([r.choices[0].message.content for r in responses if r.choices])

    async def generate_structured(self, message, response_model: Type[ModelT],
                                  request_params: RequestParams | None = None):
        response_text = await self.generate_str(message, request_params)
        # GigaChat structured output handling would go here
        return response_model.parse_raw(response_text)

    async def execute_tool_call(self, tool_call):
        try:
            tool_args = json.loads(tool_call.function.arguments)
            tool_call_request = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name=tool_call.function.name, arguments=tool_args),
            )
            result = await self.call_tool(tool_call_request)
            return Messages(
                role=MessagesRole.TOOL,
                content=[mcp_content_to_gigachat_content(c) for c in result.content],
                tool_call_id=tool_call.id
            )
        except json.JSONDecodeError as e:
            return Messages(
                role=MessagesRole.TOOL,
                content=f"JSON Error: {str(e)}",
                tool_call_id=tool_call.id
            )

    def handle_content(self, messages, message):
        messages.append(Messages(
            role=MessagesRole.ASSISTANT,
            content=message.content
        ))

    async def handle_tool_calls(self, tool_calls, messages):
        tool_responses = await self.executor.execute(
            *[self.execute_tool_call(tc) for tc in tool_calls]
        )
        messages.extend(tool_responses)

    def should_stop(self, response):
        finish_reason = response.choices[0].finish_reason if response.choices else None
        return finish_reason in ["stop", "length", "content_filter"]


class MCPGigaChatTypeConverter(ProviderToMCPConverter[Messages, Chat]):
    @classmethod
    def from_mcp_message_result(cls, result: MCPMessageResult) -> Chat:
        return Chat(
            choices=[{
                "message": {
                    "role": MessagesRole.ASSISTANT,
                    "content": result.content.text
                }
            }]
        )

    @classmethod
    def to_mcp_message_result(cls, result: Chat) -> MCPMessageResult:
        return MCPMessageResult(
            role=MessagesRole.ASSISTANT,
            content=TextContent(text=result.choices[0].message.content),
            model=result.model
        )

    @classmethod
    def from_mcp_message_param(cls, param: MCPMessageParam) -> Messages:
        return Messages(
            role=param.role,
            content=mcp_content_to_gigachat_content(param.content)
        )

    @classmethod
    def to_mcp_message_param(cls, param: Messages) -> MCPMessageParam:
        return MCPMessageParam(
            role=param.role,
            content=TextContent(text=param.content)
        )


def mcp_content_to_gigachat_content(content):
    if isinstance(content, TextContent):
        return content.text
    elif isinstance(content, (ImageContent, EmbeddedResource)):
        return str(content)  # GigaChat specific content handling
    return str(content)


def gigachat_content_to_mcp_content(content):
    return [TextContent(text=content)]