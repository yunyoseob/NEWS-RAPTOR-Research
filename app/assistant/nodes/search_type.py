from app.assistant.state import AgentState

def Search_type(state: AgentState):
    search_type = state.search_type
    if search_type == "LLM":
        return "run_prompt"
    else:
        return "retrieve"