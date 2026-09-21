import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():

    client = MultiServerMCPClient(
        {
            "commerce": {
                "transport": "sse",
                "url": "https://slavpilus-mpc-for-commerce-platforms.hf.space/gradio_api/mcp/sse",
            }
        }
    )

    tools = await client.get_tools()

    for tool in tools:
        print(tool.name)
        print(tool.description)
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())