import asyncio

from graph import graph


async def main():
    result = await graph.ainvoke(
        {
            "message": "What's the weather today?",
            "route": "",
            "response": "",
            "messages": [],
        },
        config={
            "configurable": {
                "thread_id": "test-user-1"
            }
        }
    )

    print("\n")
    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result["response"])


if __name__ == "__main__":
    asyncio.run(main())