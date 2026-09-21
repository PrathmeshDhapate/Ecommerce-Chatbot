from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.sse import sse_client


MCP_SERVER_URL = (
    "https://slavpilus-mpc-for-commerce-platforms.hf.space/"
    "gradio_api/mcp/sse"
)


@asynccontextmanager
async def connect_to_mcp():

    async with sse_client(MCP_SERVER_URL) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            yield session