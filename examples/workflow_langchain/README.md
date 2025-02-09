# MCP LangChain Agent

mcp-agent implements multi-agent workflows using [LangChain](https://github.com/langchain-ai/langchain), demonstrating integration with Sber's GigaChat models through LangChain's framework.

## Agents

1. **Triage Agent**: Routes requests to appropriate specialist agents
2. **Flight Modification Agent**: Handles flight changes through sub-agents:
   - **Flight Cancel Agent**: Processes cancellations
   - **Flight Change Agent**: Manages rescheduling
3. **Lost Baggage Agent**: Handles baggage-related inquiries

## Implementation Code

```python
from mcp_agent.workflows.langchain import LangChainIntegration
from mcp_agent.workflows.langchain.langchain_gigachat import GigaChatLangChain
from langchain_community.chat_models import GigaChat

class TriageAgent(LangChainIntegration):
    def __init__(self, credentials: str):
        llm = GigaChat(
            credentials=credentials,
            scope="GIGACHAT_API_CORP",
            temperature=0.3,
            max_tokens=2048
        )
        super().__init__(
            agent=BaseAgent(name="triage"),
            llm=llm
        )

# Initialize workflow
async def main():
    triage = TriageAgent(credentials="your-gigachat-credentials")
    await triage.agent.initialize()
    
    response = await triage.generate([{
        "role": "user",
        "content": "I need to cancel my flight to Moscow"
    }])
    
    print("Triage Response:", response)

if __name__ == "__main__":
    asyncio.run(main())