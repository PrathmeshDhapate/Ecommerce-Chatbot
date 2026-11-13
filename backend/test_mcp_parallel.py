
import asyncio
import time

from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# 1. Initialize MCP Client
# ============================================================

MCP_CLIENT = MultiServerMCPClient(
    {
        "commerce": {
            "transport": "sse",
            "url": "https://slavpilus-mpc-for-commerce-platforms.hf.space/gradio_api/mcp/sse",
        }
    }
)


# ============================================================
# 2. Test Gemini Parallel Tool Calling
# ============================================================

async def test_mcp_parallel_calls():

    print("\nLoading MCP tools...")

    start = time.perf_counter()

    tools = await MCP_CLIENT.get_tools()

    print(f"Tools loaded in {time.perf_counter() - start:.2f}s")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        streaming=False,
    )

    mcp_llm = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content="""
            You are an e-commerce assistant.

            When a user requests multiple independent operations,
            call all required tools in the same response.

            Do not wait for one tool result before requesting
            another independent operation.
            """
        ),
        HumanMessage(
            content="""
            Retrieve both the current order status and latest
            package tracking information.

            Order ID: ORD-1005-D
            Customer ID: CUST-1005

            Call get_order_status and track_package together.
            Generate exactly two tool calls in this response.
            """
        ),
    ]

    print("\nInvoking Gemini...")

    start = time.perf_counter()

    result = await mcp_llm.ainvoke(messages)

    elapsed = time.perf_counter() - start

    print("\n========== GEMINI RESULTS ==========")

    print("LLM execution time:", round(elapsed, 3), "seconds")
    print("Number of tool calls:", len(result.tool_calls))

    for call in result.tool_calls:
        print("\nTool:", call["name"])
        print("Arguments:", call["args"])

    expected_tools = {
        "get_order_status",
        "track_package",
    }

    actual_tools = {call["name"] for call in result.tool_calls}

    if len(result.tool_calls) == 2 and actual_tools == expected_tools:
        print("\nSUCCESS: Both tool calls generated.")
    else:
        print("\nWARNING: Expected two independent tool calls.")

    print("====================================")


# ============================================================
# 3. Test ToolNode Parallel Execution
# ============================================================

async def test_toolnode_parallel():

    print("\nLoading MCP tools...")

    tools = await MCP_CLIENT.get_tools()

    print("\nAvailable MCP tools:")

    for tool in tools:
        print("-", tool.name)

    tool_node = ToolNode(tools)

    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_order_status",
                "args": {
                    "order_id": "ORD-1005-D",
                    "customer_id": "CUST-1005",
                },
                "id": "test_call_1",
                "type": "tool_call",
            },
            {
                "name": "track_package",
                "args": {
                    "order_id": "ORD-1005-D",
                    "customer_id": "CUST-1005",
                },
                "id": "test_call_2",
                "type": "tool_call",
            },
        ],
    )

    print("\nExecuting both tools through ToolNode...")

    start = time.perf_counter()

    try:

        result = await tool_node.ainvoke(
            {
                "messages": [ai_message]
            },
            runtime=Runtime(),
        )

        elapsed = time.perf_counter() - start

        print("\n========== TOOLNODE RESULTS ==========")

        print("Total execution time:", round(elapsed, 3), "seconds")
        print("Number of tool results:", len(result["messages"]))

        for message in result["messages"]:

            print("\nTool:", message.name)
            print("Tool call ID:", message.tool_call_id)
            print("Result:", message.content)
            print("Status:", message.status)

        print("\n======================================")

    except Exception as e:

        print("\nToolNode execution failed!")
        print("Error:", repr(e))


# ============================================================
# 4. Main
# ============================================================

async def main():

    print("\n======================================")
    print("       MCP PARALLEL EXECUTION TEST")
    print("======================================")

    await test_mcp_parallel_calls()

    await test_toolnode_parallel()


if __name__ == "__main__":
    asyncio.run(main())
