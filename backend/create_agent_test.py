import asyncio
import time
import graph


QUERY = (
    "For order ORD-1005-D and customer cust001, "
    "tell me the current shipping status and also give me "
    "the general return policy."
)


async def main():

    print("=" * 70)
    print("CREATE_AGENT LATENCY TEST")
    print("=" * 70)

    print("\nQUERY:")
    print(QUERY)

    # Initialize MCP and create_agent
    await graph.initialize_mcp()

    print("\n" + "=" * 70)
    print("RUNNING CREATE_AGENT")
    print("=" * 70)

    # Create timing callback
    timing_callback = graph.MCPTimingCallback()

    start = time.perf_counter()

    result = await graph.MCP_AGENT.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": QUERY
                }
            ]
        },
        config={
            "callbacks": [timing_callback]
        }
    )

    total = time.perf_counter() - start

    final_message = result["messages"][-1]

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(final_message.content)

    print("\n" + "=" * 70)
    print("TOTAL TIMING")
    print("=" * 70)

    print(f"\nTotal create_agent time: {total:.3f}s")


if __name__ == "__main__":
    asyncio.run(main())