from app.vectordb.vectordb import get_vectorstore
from app.assistant.state import AgentState
from app.config import get_settings
from app.assistant import get_openai_embeddings

config = get_settings()

def RetrieveDocuments(state: AgentState) -> AgentState:
    # Search Type: RAG or RAPTOR
    search_type = state.search_type
    query = state.query
    top_k = 3
    
    embeddings= get_openai_embeddings()
    contexts = []
    metainfo = []
    embeddings_question =  embeddings.embed_query(query)

    vectorstore = None
    if search_type == "RAG":
        vectorstore = get_vectorstore(collection_name="rag_collection")
    elif search_type == "RAPTOR":
        vectorstore = get_vectorstore(collection_name="raptor_collection")

    responses = vectorstore.similarity_search_with_score_by_vector(embeddings_question, k=top_k)    
    responses = sorted(responses, key=lambda x: x[1], reverse=True)

    docs = []
    for idx in range(0, len(responses)):
        response = responses[idx]
        if response is not None:
            document = response[0] if response is not None else None
            score = response[1] if response is not None else None
            print(f"document > {document} >> score : {score}")
            if document is not None:
                doc = {}
                doc["context"] = document.page_content
                doc["metadata"] = document.metadata
                doc["metadata"]["context"]= document.page_content
                doc["metadata"]["score"]= score
                docs.append(doc)

    contexts = [doc["context"] for doc in docs]
    contexts = " , ".join(contexts)
    metainfo = [doc["metadata"] for doc in docs]

    # retriever 생성
    print(f"contexts: {contexts}")
    print(f"metainfo: {metainfo}")

    state.contexts = contexts
    state.metainfo = metainfo

    return state