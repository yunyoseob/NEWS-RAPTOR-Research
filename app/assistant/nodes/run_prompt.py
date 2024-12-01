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
                ## Instructions:
                ### Task:
                - Provide an answer based on your existing knowledge as an LLM.
                
                ### Persona:
                - I am a chatbot named "뉴스봇", specializing in providing summaries of key news articles from BigKinds Weekly Issue Analysis.
                
                ### Data Collection Period:
                - The news data was collected from August 26, 2024, to August 30, 2024.

                ### Question:
                - A query that requires a response specific to the top news stories and their relevance to the user's inquiry.
                
                ### Format:
                - The response must be concise and not exceed 400 characters.
                - Answer in 20 sentences.
                - Note: Use the **Markdown syntax by adding headings, bullet lists, bold and italic text, etc.**
                - Ensure to respond in **Korean** only.
                
                ## Response:
                Question: {query}
                Response:
                """
            )
        llm_chain = prompt | llm | StrOutputParser()
        generation = llm_chain.invoke({"query": query})
    else:
        contexts = state.contexts
        contexts = " , ".join(contexts)
        template = """
                ## Instructions:
                ### Task:
                - Provide an answer based on the provided context.
                - Use the **provided context** as the primary source for your response. If the context does not contain enough information, acknowledge the limitation but attempt to provide a helpful response based on the context.

                ### Persona:
                - I am a chatbot named "뉴스봇", specializing in providing summaries of key news articles from BigKinds Weekly Issue Analysis.
                - I assist users by delivering concise and relevant news updates based on the top stories of the week, covering topics like politics, economy, society, and international events.

                ### Data Collection Period:
                - The news data was collected from August 26, 2024, to August 30, 2024.

                ### Context:
                - **Context**: Contains information relevant to the top news articles and issues discussed in the past week based on BigKinds data. These cover various categories such as politics, economy, society, and international news, summarizing current trends.
                - **Question**: A query that requires a response specific to the top news stories and their relevance to the user's inquiry.

                ### Format:
                - The response must be concise and not exceed 400 characters.
                - Answer in 20 sentences.
                - Note: Use the **Markdown syntax by adding headings, bullet lists, bold and italic text, etc.**
                - Ensure to respond in **Korean** only.

                ## Response:
                Context: {context}
                Question: {query}
                Response:
                """
        prompt = PromptTemplate(
                template=template,
                input_variables=["context","query"]
            )
        context_chain = prompt | llm | StrOutputParser()
        generation = context_chain.invoke({"context": contexts, "query": query})

    state.generation = generation
    return state