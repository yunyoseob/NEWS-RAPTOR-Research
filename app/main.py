import os
import gradio as gr
from dotenv import load_dotenv
import openai
from app.config import get_settings

config = get_settings()

# OpenAI API Key Check
openai.api_key = config.OPENAI_API_KEY 

def get_ai_answer(message, history):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{message}"},
        ]
    )
    return response.choices[0].message['content']

chat = gr.ChatInterface(
   fn=get_ai_answer,
   theme="soft",
   examples=["오늘의 핫한 기사를 알려줘"],
   title="Korean News Search Chat Bot",
)
chat.launch()


