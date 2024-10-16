from functools import lru_cache

from langgraph.graph import END, StateGraph
from app.assistant.state import AgentState
from app.assistant.nodes import (Search_type, RetrieveDocuments, RunPrompt)
from PIL import Image as PILImage

from io import BytesIO
import nest_asyncio
from IPython.display import display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from PIL import Image as PILImage

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

if __name__ == "__main__":
    # Compile the graph
    graph = CompilingGraph()

    nest_asyncio.apply()  # Required for Jupyter Notebook to run async functions

    png_data = graph.get_graph().draw_mermaid_png(
                curve_style=CurveStyle.LINEAR,
                node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
                wrap_label_n_words=9,
                output_file_path=None,
                draw_method=MermaidDrawMethod.PYPPETEER,
                background_color="white",
                padding=10,
            )
    with open("graph.png", "wb") as f:
        f.write(png_data)  # png_data should now contain the binary image data
    # Optionally display the image using PIL
    img = PILImage.open("graph.png")
    img.show()