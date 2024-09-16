from app.vectordb.vectordb import get_vectorstore
from app.assistant.state import AgentState
from app.config import get_settings
from langchain_openai import OpenAIEmbeddings

config = get_settings()

def RetrieveDocuments(state: AgentState) -> AgentState:
    # Search Type: RAG or RAPTOR
    search_type = state.search_type
    query = state.query
    top_k = 3
    
    embeddings= OpenAIEmbeddings(model="text-embedding-3-large")
    threshold = 0.3
    contexts = []
    metainfo = []

    if search_type == "RAG":
        vectorstore = get_vectorstore(collection_name="rag_collection")
        retreiever = vectorstore.as_retriever(search_kwargs={'k':top_k})    
        docs = retreiever.invoke(query)
        contexts = [doc.page_content for doc in docs]
        metainfo = [doc.metadata for doc in docs]

    elif search_type == "RAPTOR":
        vectorstore = get_vectorstore(collection_name="raptor_collection")
        retreiever = vectorstore.as_retriever(search_kwargs={'k':top_k})    
        docs = retreiever.invoke(query)
        contexts = [doc.page_content for doc in docs]
        metainfo = [doc.metadata for doc in docs]
        """
        embeddings_question = await embeddings.embed_query(query)
        response = vectorstore.similarity_search_with_score_by_vector(embeddings_question, k=top_k)
        docs = []
        for doc, score in response:
            print(f"docs > {docs} >> score : {score} threshold :  {threshold})")
            if score > threshold:
                docs.append({"doc_text":doc.page_content, "doc_meta":doc.metadata})
        contexts = [doc["doc_text"] for doc in docs]
        contexts = " , ".join(contexts)
        """
    # retriever 생성
    print(f"contexts: {contexts}")
    print(f"metainfo: {metainfo}")

    state.contexts = contexts
    state.metainfo = metainfo

    return state