from __future__ import annotations

import asyncio
import aiohttp
import unicodedata
import re
from typing import Any, Dict, Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

import importlib.metadata
_ = importlib.metadata.version("langchain-google-genai")

# -------- Contexto --------
class Context(TypedDict):
    my_configurable_param: str


# -------- Estado --------
class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: str


# -------- Tool del clima --------
@tool
async def get_detailed_weather_weatherapi(city: str) -> str:
    """Obtiene el clima actual para una ciudad dada usando wttr.in."""
    url = f"https://wttr.in/{city}?format=3"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    return f"Error al obtener el clima: respuesta {response.status} de wttr.in"
    except asyncio.TimeoutError:
        return "Error al obtener el clima: la solicitud excedió el tiempo de espera."
    except Exception as e:
        return f"Error al obtener el clima: {str(e)}"



# -------- Tool de Ragie --------
@tool
async def search_knowledge(query: str, top_k: int = 7) -> dict:
    """Busca en la base de conocimiento usando Ragie."""
    api_key = "tnt_2MqrV4FyjTC_AzTuSG6GxrAUb1Kjp18eNzDWeTgtel0eEIMUZgBElvG"
    partition = "t-systems"

    # Normalización
    q = unicodedata.normalize('NFKC', query or "").lower().strip()
    q = ''.join(ch for ch in unicodedata.normalize('NFD', q) if unicodedata.category(ch) != 'Mn')
    q = re.sub(r"[^\w\s]", " ", q, flags=re.UNICODE)
    normalized_query = re.sub(r"\s+", " ", q).strip()

    body = {
        "query": normalized_query,
        "top_k": top_k,
        "rerank": False,
        "max_chunks_per_document": 0,
        "partition": partition,
        "recency_bias": False
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.ragie.ai/retrievals", json=body, headers=headers, timeout=30
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {
                    "success": True,
                    "results": data.get("scored_chunks", []),
                    "total": len(data.get("scored_chunks", []))
                }
            else:
                return {"success": False, "error": await resp.text()}


# -------- Modelos --------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
llm_weather = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0).bind_tools([get_detailed_weather_weatherapi])
llm_ragie = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0).bind_tools([search_knowledge])
llm_router = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)


# -------- Prompt del supervisor --------
prompt_router = ChatPromptTemplate.from_messages([
    (
        "system",
        "Eres un agente supervisor. Clasifica la intención del usuario en:\n"
        "- 'weather': si pregunta por clima, tiempo, temperatura o ciudad.\n"
        "- 'ragie': si pregunta sobre proyectos, información institucional o requiere búsqueda documental.\n"
        "- 'math': si pide cálculos o matemáticas.\n"
        "- 'default': si no encaja en ninguna categoría.\n\n"
        "Responde solo con una palabra: weather, ragie, math o default."
    ),
    ("human", "{input}")
])


# -------- Nodo: Supervisor --------
async def supervisor(state: MessageState, runtime: Runtime[Context]):
    user_input = state["messages"][-1].content

    #  Ejecutamos la llamada al LLM en un hilo separado
    resp = await asyncio.to_thread(
        lambda: llm_router.invoke(prompt_router.format_messages(input=user_input))
    )

    route = resp.content.strip().lower()
    if route not in ["weather", "ragie", "math"]:
        route = "default"
    return {"route": route}

# -------- Nodo: Agente general --------
async def call_model(state: MessageState, runtime: Runtime[Context]) -> Dict[str, Any]:
    response = await asyncio.to_thread(
        lambda: llm.invoke(state["messages"])
    )
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }


#  -------- Nodo: Agente del clima con ejecución manual de la tool --------
async def weather_agent(state: MessageState, runtime: Runtime[Context]) -> Dict[str, Any]:
    # 1 Ejecutamos la llamada al LLM en un hilo separado para evitar bloqueos
    response = await asyncio.to_thread(
        lambda: llm_weather.invoke(state["messages"])
    )
    print("Weather agent raw response:", response)

    # 2️ Si hay tool_calls, las ejecutamos manualmente
    if response.tool_calls:
        results = []
        for tool_call in response.tool_calls:
            if tool_call["name"] == "get_detailed_weather_weatherapi":
                args = tool_call["args"]
                result = await get_detailed_weather_weatherapi.ainvoke(args)
                results.append(result)

        return {
            "messages": state["messages"] + [AIMessage(content="\n".join(results))]
        }

    # 3️ Si no hay tool_calls, devolvemos el contenido normal
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }



#  -------- Nodo: Agente Ragie --------
async def ragie_agent(state: MessageState, runtime: Runtime[Context]) -> Dict[str, Any]:
    response = await asyncio.to_thread(
        lambda: llm_ragie.invoke(state["messages"])
    )
    print("Ragie agent raw response:", response)

    if response.tool_calls:
        results = []
        for tool_call in response.tool_calls:
            if tool_call["name"] == "search_knowledge":
                args = tool_call["args"]
                tool_result = await search_knowledge.ainvoke(args)
                if tool_result.get("success"):
                    chunks = tool_result.get("results", [])
                    textos = [c.get("text", "") for c in chunks[:3]]
                    results.append("\n\n".join(textos))
                else:
                    results.append(f"Error en Ragie: {tool_result.get('error')}")

        return {
            "messages": state["messages"] + [AIMessage(content="\n".join(results))]
        }

    return {
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }


# -------- Construcción del grafo --------
graph = (
    StateGraph(MessageState, context_schema=Context)
    .add_node("supervisor", supervisor)
    .add_node("call_model", call_model)
    .add_node("weather_agent", weather_agent)
    .add_node("ragie_agent", ragie_agent)  # 🟡 Nuevo nodo
    .add_conditional_edges("supervisor", lambda state: state["route"], {
        "weather": "weather_agent",
        "ragie": "ragie_agent",
        "math": "call_model",
        "default": "call_model",
    })
    .add_edge("weather_agent", END)
    .add_edge("ragie_agent", END)
    .add_edge("call_model", END)
    .add_edge(START, "supervisor")
    .compile(name="LangGraph - Supervisor + Clima + Ragie")
)
