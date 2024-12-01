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

def invoke_query(query:str, search_type:str, chat_history):
    print(f"query : {query}")
    print(f"search_type : {search_type}")    
    metainfo = None

    # 예외 처리: query가 없을 때
    if query is None or query.strip() == "":
        return "query는 필수입니다.", [], []

    # 예외 처리: search_type이 없을 때
    if search_type is None or search_type.strip() == "":
        return "Search Type은 필수입니다.", [], []
    
    
    state = graph.invoke({"query": query, "search_type": search_type})
    contexts = []

    if search_type == "LLM":
        generation = state["generation"]
    else:
        generation = state["generation"]
        metainfo = state["metainfo"]

     # metainfo가 비어 있지 않다면, 키-값 쌍을 데이터 프레임 형태로 변환
    if metainfo is not None and len(metainfo) > 0:
        metainfo_list = []
        for idx, item in enumerate(metainfo, start=1):  # index를 1부터 시작
            context = item.get("context", "")
            score = item.get("score", "")
            contexts.append([idx, context, score])
            for k, v in item.items():
                if k not in ["context", "score"]:
                    metainfo_list.append([idx, k, v])  # index, key, value로 리스트에 추가
    else:
        metainfo_list = [[0, "No data", "No data"]]

    # 채팅 기록 업데이트
    chat_history.append((query, generation))

    return chat_history, metainfo_list, contexts

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="gray")) as chat:
    gr.Markdown("<h3 style='text-align: center;'>빅카인즈 주간이슈 뉴스 데이터를 활용한 LLM 기반 한국어 뉴스 응답 시스템 개발</h3>")

    with gr.Row():
        with gr.Column(scale=4):
            query = gr.Textbox(label="Enter your message", placeholder="빅카인즈 주간 이슈에 대해 궁금한 것을 물어보세요.", lines=1)
            search_type = gr.Radio(
                choices=["LLM", "RAG", "RAPTOR"], 
                label="search_type", 
                value="RAPTOR",
                info="Select the method to LLM Prompt"
            )
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")
            
            contexts = gr.Dataframe(headers=["Index", "Context", "Score"], label="Contexts", interactive=True)
            metainfo = gr.Dataframe(headers=["Index", "Key", "Value"], label="Metainfo", interactive=True)
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(
                label="Bigkinds News Chatbot",
                elem_id="chatbot",
                height=700
            )

    submit_btn.click(invoke_query, inputs=[query, search_type, chatbot], outputs=[chatbot, metainfo, contexts])

    # clear 버튼 클릭 시 채팅 기록 초기화
    clear_btn.click(lambda: None, None, [chatbot])

if __name__ == "__main__":
    chat.launch()