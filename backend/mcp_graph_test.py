import asyncio

from graph import graph


async def test(question):

    result = await graph.ainvoke(
        {
            "message": question,
            "route": "",
            "response": ""
        }
    )

    print("\nQUESTION:")
    print(question)

    print("\nROUTE:")
    print(result["route"])

    print("\nRESPONSE:")
    print(result["response"])


async def main():

    await test(
        "What is the status of order ORD-1001-D for customer CUST-001?"
    )

    await test(
        "What is your return policy?"
    )


if __name__ == "__main__":
    asyncio.run(main())