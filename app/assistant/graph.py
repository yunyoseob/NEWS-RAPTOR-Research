from functools import lru_cache

from langgraph.graph import END, StateGraph
from app.assistant.state import AgentState
from app.assistant.nodes import (Search_type, RetrieveDocuments, RunPrompt)

@lru_cache
def CompilingGraph():
    graph = StateGraph(AgentState)

    # Docs Search
    graph.add_node("retrieve", RetrieveDocuments)  # retrieve
    graph.add_node("run_prompt", RunPrompt)  # generate_prompt

    graph.set_conditional_entry_point(Search_type,
        path_map={
            "run_prompt": "run_prompt",
            "retrieve": "retrieve",
        })

    # Docs Search Graph
    graph.add_edge("retrieve", "run_prompt")
    graph.add_edge("run_prompt", END)
    app = graph.compile()
    
    return app
