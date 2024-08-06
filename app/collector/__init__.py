from dotenv import load_dotenv
import os
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