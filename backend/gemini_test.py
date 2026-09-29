import asyncio
import time

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


async def main():

    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )

    start = time.perf_counter()

    response = await model.ainvoke(
        "Explain in one sentence what an order return policy is."
    )

    end = time.perf_counter()

    print(f"Gemini normal call: {end - start:.3f}s")
    print(response.content)


asyncio.run(main())