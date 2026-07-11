from google import genai
from google.genai import types
from google.genai import errors as genai_errors
import os
from dotenv import load_dotenv
import time
import asyncio

load_dotenv()

DEFAULT_FALLBACKS = os.getenv(
    "GEMINI_MODEL_FALLBACKS",
    "gemini-2.5-flash-lite,gemini-2.5-flash,gemini-2.0-flash,gemini-2.0-flash-lite",
)


def interactice_chatbot(query: str, model: str = "gemini-flash-lite-latest"):
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_1"))
        response = client.models.generate_content_stream(model=model, contents=query)
        for chunk in response:
            print(chunk.text)
    except genai_errors.APIError as e:
        print(f"API Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    while True:
        query = input("Enter your message: ")
        start_time = time.perf_counter()
        print("Processing...")
        interactice_chatbot(query)
        end_time = time.perf_counter()
        print(f"Response time: {end_time - start_time:.2f} seconds")
        if query.lower() in ["exit", "quit"]:
            print("Exiting chatbot.")
            break
