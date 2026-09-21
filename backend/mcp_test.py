import asyncio

from mcp_connection import connect_to_mcp


async def main():

    async with connect_to_mcp() as session:

        result = await session.call_tool(
            "get_order_status",
            {
                "order_id": "ORD-1001-D",
                "customer_id": "CUST-001"
            }
        )

        print(result)


if __name__ == "__main__":
    asyncio.run(main())