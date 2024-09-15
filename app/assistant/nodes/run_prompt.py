from langchain_core.prompts import (PromptTemplate, ChatPromptTemplate)
from app.assistant.state import AgentState
from app.assistant import get_chatllm_openai
from langchain_core.output_parsers import StrOutputParser

def RunPrompt(state: AgentState) -> AgentState:
    search_type = state.search_type
    query = state.query

    llm = get_chatllm_openai()

    if search_type == "LLM":
        prompt = ChatPromptTemplate.from_template(
                """
                    사용자의 question에 답변해줘. 
                    Question: {query}
                """
            )
        llm_chain = prompt | llm | StrOutputParser()
        generation = llm_chain.invoke({"query": query})
    else:
        contexts = state.contexts
        contexts = " , ".join(contexts)
        template = """
            아래의 <context> 내용만 이용하여 사용자의 question에 답변해줘. 모르면 알지 못한다고 말해주고 context를 봤다는 언급은 하지 말아줘.
            
            <context>
                {context}
            </context>
            
            Question: 
                {query}
            """
        prompt = PromptTemplate(
                template=template,
                input_variables=["context","query"]
            )
        context_chain = prompt | llm | StrOutputParser()
        generation = context_chain.invoke({"context": contexts, "query": query})

    state.generation = generation
    return state