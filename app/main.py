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

    if not search_type:
        return "Search Type은 필수입니다.", []

    if search_type == "LLM":
        generation = state["generation"]
        metainfo = []
    else:
        metainfo = state["metainfo"]
        generation = state["generation"]

    print(f"generation: {generation}")
     # metainfo가 비어 있지 않다면, 키-값 쌍을 데이터 프레임 형태로 변환
    if metainfo is not None and len(metainfo) > 0:
        metainfo_list = []
        for idx, item in enumerate(metainfo, start=1):  # index를 1부터 시작
            for k, v in item.items():
                metainfo_list.append([idx, k, v])  # index, key, value로 리스트에 추가
    else:
        metainfo_list = [[0, "No data", "No data"]]

    return generation, metainfo_list

chat = gr.Interface(
    fn=invoke_query, 
    inputs=[
        gr.Textbox(label="query", placeholder="빅카인즈 주간 이슈에 대해 궁금한 것을 물어보세요."), 
        gr.Radio(
            choices=["LLM", "RAG", "RAPTOR"], 
            label="search_type", 
            info="Select the method to LLM Prompt"
        ),
    ],
    outputs=[
                gr.Textbox(label="Response", lines=15),
                gr.Dataframe(headers=["Index", "Key", "Value"], label="Metainfo", interactive=False)
            ]
)
if __name__ == "__main__":
    chat.launch()