import os
import gradio as gr
from dotenv import load_dotenv
import openai

# load .env
# Write Your API Key In .env
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# OpenAI API Key Check
if openai_api_key:
    openai.api_key = openai_api_key 
else:
    print("Error: OPENAI_API_KEY is not set in the .env file")
    exit(1)

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


