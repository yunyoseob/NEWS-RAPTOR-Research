import os
import gradio as gr
from app.config import get_settings

config = get_settings()

try:
    from app.assistant.graph import CompilingGraph
    graph = CompilingGraph()
except Exception as e:
    print(f"FAILED :  Graph Compiling - {e}")
else:
    print("LOADED: Graph Compiling")

def invoke_query(query:str, search_type:str):
    print(f"query : {query}")
    print(f"search_type : {search_type}")    
    state = graph.invoke({"query": query, "search_type": search_type})

    if search_type == "LLM":
        generation = state["generation"]
    else:
        metainfo = state["metainfo"]
        generation = state["generation"]

    print(f"generation: {generation}")
    return generation

chat = gr.Interface(
    fn=invoke_query, 
    inputs=[
        gr.Textbox(label="query", placeholder="빅카인즈 주간 이슈에 대해 궁금한 것을 물어보세요."),  # Textbox for query
        gr.Radio(
            choices=["LLM", "RAG", "RAPTOR"], 
            label="search_type", 
            info="Select the method to LLM Prompt"
        ),
    ],
    outputs=gr.Textbox(label="Response", lines=25)  
)
if __name__ == "__main__":
    chat.launch()