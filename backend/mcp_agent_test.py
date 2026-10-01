
import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient


load_dotenv()


MCP_URL = (
    "https://slavpilus-mpc-for-commerce-platforms.hf.space"
    "/gradio_api/mcp/sse"
)


SYSTEM_PROMPT = """
You are a commerce support agent.

You have access to MCP tools for handling commerce-related requests.

IMPORTANT RULES:

1. Use an MCP tool when the user's request can be handled by one
   of the available tools.

2. Never invent, guess, or fabricate customer IDs.

3. If the user explicitly provides a customer ID, use exactly that
   customer ID.

4. If a tool requires a customer ID but the user has not provided one,
   do not invent one such as:
   - "unknown"
   - "default_customer"
   - "CUST-001"
   - any other guessed value.

5. For unsupported requests, do not call an unrelated MCP tool.
   Clearly explain that you can only help with commerce-related
   requests.

6. Always use the actual MCP tool result when generating the final
   answer.

7. Never fabricate information that was not returned by an MCP tool.
"""


async def main():

    # =========================================================
    # 1. Connect to MCP server
    # =========================================================

    print("\nConnecting to MCP server...")

    client = MultiServerMCPClient(
        {
            "commerce": {
                "transport": "sse",
                "url": MCP_URL,
            }
        }
    )

    # =========================================================
    # 2. Load MCP tools
    # =========================================================

    print("Loading MCP tools...")

    tools = await client.get_tools()

    print("\nAVAILABLE MCP TOOLS:")

    for tool in tools:
        print("-", tool.name)

    print("\nTotal tools:", len(tools))

    # =========================================================
    # 3. Create Gemini LLM
    # =========================================================

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )

    # =========================================================
    # 4. Create MCP agent
    # =========================================================

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT
    )

    # =========================================================
    # 5. Test queries
    # =========================================================

    test_queries = [

        # -----------------------------------------------------
        # Test 1: Customer ID explicitly provided
        # -----------------------------------------------------

        "What is the status of order ORD-123 for customer CUST-001?",

        # -----------------------------------------------------
        # Test 2: Customer ID missing
        # -----------------------------------------------------

        "What is the status of order ORD-123?",

        # -----------------------------------------------------
        # Test 3: Action request
        # -----------------------------------------------------

        "Cancel order ORD-123",

        # -----------------------------------------------------
        # Test 4: Unsupported request
        # -----------------------------------------------------

        "What's the weather today?",
    ]

    # =========================================================
    # 6. Run tests
    # =========================================================

    for query in test_queries:

        print("\n")
        print("=" * 70)
        print("USER QUERY:")
        print(query)
        print("=" * 70)

        try:

            result = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": query,
                        }
                    ]
                }
            )

            messages = result["messages"]

            print("\nAGENT MESSAGES:")

            for message in messages:

                print("\n---")

                print(
                    "TYPE:",
                    type(message).__name__
                )

                if hasattr(message, "name"):
                    print(
                        "NAME:",
                        message.name
                    )

                if hasattr(message, "tool_calls"):

                    if message.tool_calls:

                        print(
                            "TOOL CALLS:",
                            message.tool_calls
                        )

                print(
                    "CONTENT:",
                    message.content
                )

            print("\nFINAL ANSWER:")

            final_message = messages[-1]

            print(final_message.content)

        except Exception as e:

            print("\nERROR:")

            print(
                type(e).__name__,
                e
            )


if __name__ == "__main__":
    asyncio.run(main())
