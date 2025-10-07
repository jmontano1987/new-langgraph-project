"""LangGraph single-node graph template con invocación a un LLM con streaming."""

from __future__ import annotations

from dataclasses import dataclass
from langgraph.graph.message import add_messages
from typing import Annotated, AsyncGenerator, Dict, Any

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI


# -------- Contexto --------
class Context(TypedDict):
    my_configurable_param: str


# -------- Estado --------
@dataclass
class State:
    messages: Annotated[list, add_messages]


# -------- Modelo LLM --------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)


# -------- Nodo que invoca el modelo (STREAMING) --------
async def call_model(
    state: State,
    runtime: Runtime[Context]
) -> AsyncGenerator[Dict[str, Any], None]:
    """Invoca el LLM en streaming y va emitiendo tokens a medida que llegan."""
    # Usamos astream para recibir chunks incrementales
    async for chunk in llm.astream(state.messages):
        if chunk.content:  # Filtramos tokens vacíos
            yield {
                "messages": [
                    {"role": "assistant", "content": chunk.content}
                ]
            }


# -------- Definición del grafo --------
graph = (
    StateGraph(State, context_schema=Context)
    .add_node(call_model)
    .add_edge("__start__", "call_model")
    .compile(name="agent")   # 👈 este nombre debe coincidir con el usado en el SDK
)
