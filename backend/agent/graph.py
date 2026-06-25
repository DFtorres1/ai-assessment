from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from agent.state import AgentState


def _route_after_router(state: AgentState) -> str:
    if state.get("intent") == "out_of_scope":
        return "out_of_scope"
    return "in_scope"


def _route_after_retrieve(state: AgentState) -> str:
    intent = state.get("intent", "")
    if intent == "holiday_timing":
        return "needs_holidays"
    if state.get("route_to_fallback", False):
        return "low_confidence"
    if not state.get("retrieved_chunks"):
        return "low_confidence"
    return "confident"


async def create_graph(chroma_client: Any) -> Any:
    import agent.guards.input_guard as input_guard_mod
    import agent.guards.output_guard as output_guard_mod
    from adapters.embedding.sentence_transformer import SentenceTransformerAdapter
    from adapters.holidays.nager import NagerHolidaysAdapter
    from adapters.llm.anthropic import AnthropicAdapter
    from adapters.vector_store.chromadb import ChromaDBVectorStoreAdapter
    from agent.nodes import answer as answer_mod
    from agent.nodes import fallback as fallback_mod
    from agent.nodes import holidays as holidays_mod
    from agent.nodes import router as router_mod
    from agent.nodes.holidays import holidays_tool_node
    from agent.nodes.retrieve import make_retrieve_node
    from config import settings

    # ── Create adapters ────────────────────────────────────────────────────────
    embed_adapter = SentenceTransformerAdapter(model_name="all-MiniLM-L6-v2")

    haiku = AnthropicAdapter(model=settings.haiku_model, temperature=0.2, max_tokens=1024)
    sonnet = AnthropicAdapter(model=settings.sonnet_model, temperature=0.2, max_tokens=2048)

    vector_store = ChromaDBVectorStoreAdapter(
        chroma_client=chroma_client,
        embed_fn=embed_adapter.chroma_ef,
        collection_name=settings.chroma_collection_name,
    )

    holidays_adapter = NagerHolidaysAdapter()

    # ── Inject into domain modules ─────────────────────────────────────────────
    input_guard_mod.input_guard_llm = haiku
    router_mod.router_llm = haiku
    router_mod.expander_llm = haiku
    answer_mod.haiku_llm = haiku
    answer_mod.sonnet_llm = sonnet
    fallback_mod.fallback_llm = haiku
    output_guard_mod.grounding_llm = haiku
    holidays_mod._tool = holidays_adapter

    answer_mod._embed_fn = embed_adapter.embed

    from agent.nodes.router import _EXAMPLE_BANK_ENTRIES, ExampleBank

    router_mod._example_bank = ExampleBank(_EXAMPLE_BANK_ENTRIES, embed_adapter.embed)

    retrieve_node = make_retrieve_node(vector_store)

    # ── Build graph ────────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    graph.add_node("router", router_mod.router_node)
    graph.add_node("query_expander", router_mod.query_expander_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("holidays_tool", holidays_tool_node)
    graph.add_node("answer", answer_mod.answer_node)
    graph.add_node("fallback", fallback_mod.fallback_node)
    graph.add_node("rejection", fallback_mod.rejection_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {"in_scope": "query_expander", "out_of_scope": "rejection"},
    )
    graph.add_edge("query_expander", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {
            "confident": "answer",
            "needs_holidays": "holidays_tool",
            "low_confidence": "fallback",
        },
    )
    graph.add_edge("holidays_tool", "answer")
    graph.add_edge("answer", END)
    graph.add_edge("fallback", END)
    graph.add_edge("rejection", END)

    return graph.compile()
