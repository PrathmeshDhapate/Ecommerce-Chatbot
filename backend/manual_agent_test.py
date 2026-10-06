import asyncio
import time
from typing import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import AsyncCallbackHandler

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_mcp_adapters.client import MultiServerMCPClient


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

QUERY = (
    "For order ORD-1005-D and customer cust001, "
    "tell me the current shipping status and also give me "
    "the general return policy."
)

MCP_URL = (
    "https://slavpilus-mpc-for-commerce-platforms.hf.space/"
    "gradio_api/mcp/sse"
)


# ============================================================
# TIMING CALLBACK
# ============================================================

class TimingCallback(AsyncCallbackHandler):

    def __init__(self):
        super().__init__()

        self.llm_count = 0
        self.tool_count = 0

        self.llm_start_times = {}
        self.tool_start_times = {}

        self.llm_times = []
        self.tool_times = []

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    async def on_llm_start(
        self,
        serialized,
        prompts,
        **kwargs
    ):
        self.llm_count += 1

        call_id = self.llm_count

        self.llm_start_times[call_id] = time.perf_counter()

        print(
            f"\n>>> LLM #{call_id} START"
        )

    async def on_llm_end(
        self,
        response,
        **kwargs
    ):
        if not self.llm_start_times:
            return

        call_id = max(self.llm_start_times)

        elapsed = (
            time.perf_counter()
            - self.llm_start_times.pop(call_id)
        )

        self.llm_times.append(elapsed)

        print(
            f">>> LLM #{call_id} END: "
            f"{elapsed:.3f}s"
        )

    # --------------------------------------------------------
    # TOOLS
    # --------------------------------------------------------

    async def on_tool_start(
        self,
        serialized,
        input_str,
        **kwargs
    ):
        self.tool_count += 1

        call_id = self.tool_count

        tool_name = serialized.get(
            "name",
            "unknown_tool"
        )

        self.tool_start_times[call_id] = (
            time.perf_counter()
        )

        print(
            f"\n>>> TOOL #{call_id} START: "
            f"{tool_name}"
        )

    async def on_tool_end(
        self,
        output,
        **kwargs
    ):
        if not self.tool_start_times:
            return

        call_id = min(self.tool_start_times)

        elapsed = (
            time.perf_counter()
            - self.tool_start_times.pop(call_id)
        )

        self.tool_times.append(elapsed)

        print(
            f">>> TOOL #{call_id} END: "
            f"{elapsed:.3f}s"
        )


# ============================================================
# LANGGRAPH STATE
# ============================================================

class ManualState(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# MANUAL LANGGRAPH
# ============================================================

async def run_manual_graph(llm, tools, query):

    print("\n")
    print("=" * 70)
    print("MANUAL LANGGRAPH TEST")
    print("=" * 70)

    callback = TimingCallback()

    # --------------------------------------------------------
    # Bind MCP tools to Gemini
    # --------------------------------------------------------

    print("\nBinding tools to Gemini...")

    model_with_tools = llm.bind_tools(tools)

    # --------------------------------------------------------
    # LLM NODE
    # --------------------------------------------------------

    async def call_model(state):

        node_start = time.perf_counter()

        print("\n--- LLM NODE START ---")

        response = await model_with_tools.ainvoke(
            state["messages"],
            config={
                "callbacks": [callback]
            }
        )

        node_time = (
            time.perf_counter()
            - node_start
        )

        print(
            f"--- LLM NODE END: "
            f"{node_time:.3f}s ---"
        )

        return {
            "messages": [response]
        }

    # --------------------------------------------------------
    # TOOL NODE
    # --------------------------------------------------------

    tool_node = ToolNode(tools)

    async def call_tools(state):
        start = time.perf_counter()

        result = await tool_node.ainvoke(state)

        elapsed = time.perf_counter() - start
        print(f"ToolNode: {elapsed:.3f}s")

        return result

    # --------------------------------------------------------
    # BUILD LANGGRAPH
    # --------------------------------------------------------

    graph = StateGraph(ManualState)

    graph.add_node(
        "llm",
        call_model
    )

    graph.add_node(
        "tools",
        call_tools
    )

    # START → LLM

    graph.add_edge(
        START,
        "llm"
    )

    # --------------------------------------------------------
    # LLM DECIDES:
    #
    # tool call → tools
    # final answer → END
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "llm",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        }
    )

    # --------------------------------------------------------
    # TOOLS → LLM
    #
    # After MCP tools finish, Gemini gets their results
    # and decides whether another tool is needed or whether
    # it can produce the final answer.
    # --------------------------------------------------------

    graph.add_edge(
        "tools",
        "llm"
    )

    # --------------------------------------------------------
    # COMPILE
    # --------------------------------------------------------

    manual_graph = graph.compile()

    # --------------------------------------------------------
    # EXECUTE
    # --------------------------------------------------------

    print("\nExecuting manual LangGraph...")

    total_start = time.perf_counter()

    result = await manual_graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=query
                )
            ]
        }
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    final_message = result["messages"][-1]

    print("\n")
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(final_message.content)

    # --------------------------------------------------------
    # TIMING SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal time: "
        f"{total_time:.3f}s"
    )

    print(
        f"\nLLM calls: "
        f"{callback.llm_count}"
    )

    print(
        "LLM times: "
        f"{[round(x, 3) for x in callback.llm_times]}"
    )

    print(
        f"\nTool calls: "
        f"{callback.tool_count}"
    )

    print(
        "Tool times: "
        f"{[round(x, 3) for x in callback.tool_times]}"
    )

    # --------------------------------------------------------
    # BREAKDOWN
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("LATENCY BREAKDOWN")
    print("=" * 70)

    llm_total = sum(
        callback.llm_times
    )

    tool_total = sum(
        callback.tool_times
    )

    print(
        f"\nLLM total:   "
        f"{llm_total:.3f}s"
    )

    print(
        f"Tool total:  "
        f"{tool_total:.3f}s"
    )

    print(
        f"Other:       "
        f"{max(0, total_time - llm_total - tool_total):.3f}s"
    )

    return total_time


# ============================================================
# MAIN
# ============================================================

async def main():

    print("\n")
    print("=" * 70)
    print("MANUAL LANGGRAPH LATENCY TEST")
    print("=" * 70)

    print("\nQUERY:")
    print(QUERY)

    # --------------------------------------------------------
    # MCP CLIENT
    # --------------------------------------------------------

    print("\nLoading MCP tools...")

    client = MultiServerMCPClient(
        {
            "commerce": {
                "transport": "sse",
                "url": MCP_URL,
            }
        }
    )

    tools_start = time.perf_counter()

    tools = await client.get_tools()

    tools_time = (
        time.perf_counter()
        - tools_start
    )

    print(
        f"\nMCP tools loaded in: "
        f"{tools_time:.3f}s"
    )

    print(
        f"Number of tools: "
        f"{len(tools)}"
    )

    print("\nAvailable tools:")

    for tool in tools:
        print(
            f"- {tool.name}"
        )

    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )

    # --------------------------------------------------------
    # RUN MANUAL GRAPH
    # --------------------------------------------------------

    manual_time = await run_manual_graph(
        llm,
        tools,
        QUERY
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    print(
        f"\nManual LangGraph total: "
        f"{manual_time:.3f}s"
    )

    print(
        "\nCompare this with the create_agent "
        "timing from your existing project logs."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())