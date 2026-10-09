import asyncio

from backend.graph import graph, initialize_mcp


TEST_CASE = {
    "query": "What is the status of order ORD-1005-D? My customer ID is cust005.",
    "expected_route": "mcp"
}


async def run_test():
    result = await graph.ainvoke(
        {
            "message": TEST_CASE["query"],
            "route": "",
            "response": "",
            "messages": []
        },
        config={
            "configurable": {
                "thread_id": "eval-routing-01"
            }
        }
    )

    actual_route = result["route"]

    print("\n" + "=" * 50)
    print("ROUTING EVALUATION")
    print("=" * 50)

    print(f"Expected route: {TEST_CASE['expected_route']}")
    print(f"Actual route:   {actual_route}")

    if actual_route == TEST_CASE["expected_route"]:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")


if __name__ == "__main__":
    asyncio.run(run_test())