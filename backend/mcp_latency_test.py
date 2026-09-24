import asyncio
import time

from langchain_mcp_adapters.client import MultiServerMCPClient


MCP_URL = (
    "https://slavpilus-mpc-for-commerce-platforms.hf.space"
    "/gradio_api/mcp/sse"
)


async def main():

    client = MultiServerMCPClient(
        {
            "commerce": {
                "transport": "sse",
                "url": MCP_URL,
            }
        }
    )

    print("Loading MCP tools...")

    start = time.perf_counter()

    tools = await client.get_tools()

    end = time.perf_counter()

    print(f"get_tools(): {end - start:.3f}s")

    print("\nAvailable tools:")

    for tool in tools:
        print(tool.name)

    # Find the order status tool
    tool = next(
        tool
        for tool in tools
        if tool.name == "get_order_status"
    )

    print("\nCalling get_order_status...")

    start = time.perf_counter()

    result = await tool.ainvoke(
        {
            "order_id": "ORD-1001",
            "customer_id": "CUST-001",
        }
    )

    end = time.perf_counter()

    print(f"MCP tool execution: {end - start:.3f}s")

    print("\nResult:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())