from typing import TypedDict
from typing_extensions import Annotated
import operator

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv
from rag import retrieve_faq

load_dotenv()


class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    message: str
    route: str
    response: str


def extract_text(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )

    return str(content)


async def ask_mcp_agent(messages):

    client = MultiServerMCPClient(
        {
            "commerce": {
                "transport": "sse",
                "url": "https://slavpilus-mpc-for-commerce-platforms.hf.space/gradio_api/mcp/sse",
            }
        }
    )

    # Get MCP tools
    tools = await client.get_tools()

    print("\nAVAILABLE TOOLS:")
    for tool in tools:
        print(tool.name)

    # Create Gemini model
    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash"
    )

    # Give Gemini information about the available tools
    model_with_tools = model.bind_tools(tools)

    # Create messages for Gemini
    mcp_messages = [
        SystemMessage(
            content=(
                "You are a commerce assistant. "
                "Never invent, guess, or assume values for tool arguments. "
                "Only use argument values that were explicitly provided by "
                "the user or are available in the conversation history. "
                "If a required tool argument is missing, ask the user for it "
                "instead of calling the tool with an invented value.\n\n"
                "If the request cannot be answered using any available tool, "
                "respond with exactly: UNSUPPORTED"
            )
        ),
        HumanMessage(
            content=(
                f"User request: {messages[-1].content}"
            )
        )
    ]

    # --------------------------------
    # First call to Gemini
    # --------------------------------

    response = await model_with_tools.ainvoke(mcp_messages)

    print("\nFIRST MODEL RESPONSE:")
    print(response)

    print("\nTOOL CALLS:")
    print(response.tool_calls)

    # --------------------------------
    # Tool calling loop
    # --------------------------------

    while response.tool_calls:

        # Add Gemini's response containing tool calls
        mcp_messages.append(response)

        # Handle all requested tools
        for tool_call in response.tool_calls:

            print("\nTOOL REQUESTED:")
            print(tool_call)

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print("\nTOOL NAME:")
            print(tool_name)

            print("\nTOOL ARGUMENTS:")
            print(tool_args)

            # Find the requested tool
            tool = next(
                tool for tool in tools
                if tool.name == tool_name
            )

            # Execute the MCP tool
            tool_result = await tool.ainvoke(tool_args)

            print("\nTOOL RESULT:")
            print(tool_result)

            # Add tool result to conversation
            mcp_messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                )
            )

        # --------------------------------
        # Send tool result back to Gemini
        # --------------------------------

        response = await model_with_tools.ainvoke(mcp_messages)

        print("\nMODEL RESPONSE AFTER TOOL:")
        print(response)

        print("\nTOOL CALLS AFTER TOOL:")
        print(response.tool_calls)

    # --------------------------------
    # Final response
    # --------------------------------

    print("\nFINAL RESPONSE:")
    print(response.content)

    return extract_text(response.content)


def add_user_message(state: State):

    return {
        "messages": [
            HumanMessage(content=state["message"])
        ]
    }


def rag_node(state: State):

    answer = retrieve_faq(state["message"])

    if answer is not None:
        return {
            "route": "faq_found",
            "response": answer,
            "messages": [
                AIMessage(content=answer)
            ]
        }

    return {
        "route": "mcp",
    }


async def mcp_node(state: State):

    answer = await ask_mcp_agent(state["messages"])

    if answer.strip() == "UNSUPPORTED":
        return {
            "route": "unsupported",
        }

    return {
        "route": "mcp_found",
        "response": answer,
        "messages": [
            AIMessage(content=answer)
        ]
    }


def unsupported_node(state: State):

    answer = (
        "Sorry, I can't help with that request. "
        "I can help with commerce-related questions such as "
        "orders, returns, shipping, payments, and related FAQs."
    )

    return {
        "response": answer,
        "messages": [
            AIMessage(content=answer)
        ]
    }


def route_after_rag(state: State):
    return state["route"]


def route_after_mcp(state: State):
    return state["route"]


graph_builder = StateGraph(State)

graph_builder.add_node("add_message", add_user_message)
graph_builder.add_node("rag", rag_node)
graph_builder.add_node("mcp", mcp_node)
graph_builder.add_node("unsupported", unsupported_node)

graph_builder.add_edge(START, "add_message")
graph_builder.add_edge("add_message", "rag")

graph_builder.add_conditional_edges(
    "rag",
    route_after_rag,
    {
        "faq_found": END,
        "mcp": "mcp"
    }
)

graph_builder.add_conditional_edges(
    "mcp",
    route_after_mcp,
    {
        "mcp_found": END,
        "unsupported": "unsupported"
    }
)

graph_builder.add_edge("unsupported", END)

checkpointer = InMemorySaver()

graph = graph_builder.compile(
    checkpointer=checkpointer
)