import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()


async def main():

    # 1. Connect to MCP server
    client = MultiServerMCPClient(
        {
            "commerce": {
                "transport": "sse",
                "url": "https://slavpilus-mpc-for-commerce-platforms.hf.space/gradio_api/mcp/sse",
            }
        }
    )

    # 2. Get MCP tools as LangChain tools
    tools = await client.get_tools()

    # 3. Create LLM
    model = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash"
    )

    # 4. Create LangGraph agent
    agent = create_agent(
        model,
        tools
    )

    # 5. Ask the agent a question
    result = await agent.ainvoke(
        {
            "messages": [
                (
                    "user",
                    "What is the status of order ORD-1001-D for customer CUST-001?"
                )
            ]
        }
    )

    # 6. Print the final conversation
    for message in result["messages"]:
        print(message)


if __name__ == "__main__":
    asyncio.run(main())