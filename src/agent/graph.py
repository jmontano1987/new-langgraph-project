"""LangGraph single-node graph template con invocación a un LLM."""

from __future__ import annotations

from dataclasses import dataclass
from langgraph.graph.message import add_messages
from typing import Any, Dict, Annotated

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
    #changeme: str = "Hola, esto es un ejemplo."
    messages: Annotated[list, add_messages]


# -------- Modelo LLM --------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)


# -------- Nodo que invoca el modelo --------
async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Invoca el LLM con el contenido de state.changeme."""
    response = llm.invoke(state.messages)
    return {
        "messages": [
            {"role": "assistant", "content": response.content}
        ]
    }

# -------- Definición del grafo --------
graph = (
    StateGraph(State, context_schema=Context)
    .add_node(call_model)
    .add_edge("__start__", "call_model")
    .compile(name="LLM Graph")
)
