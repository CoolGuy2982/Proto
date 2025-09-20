import os
import json
import logging
import asyncio
import re
from typing import List, Dict, AsyncGenerator
import base64
import io

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

# Correct SDK imports as per the latest documentation
from google import genai
from google.genai import types

# --- Configuration and Setup ---
logging.basicConfig(level=logging.INFO)
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it.")

# As per the new documentation, instantiate a client.
# It automatically picks up the API key from the environment variable.
client = genai.Client()

# --- System Prompts for Gemini ---
TRIAGE_SYSTEM_PROMPT = """
You're going to get a series of prompts that might include pictures; you need to decide whether it's an appropriate time to chime in and help the user or not.
Respond in this format
[["call": "YES", "reason"]] or  [["call": "NO", "reason"]]
depending on whether you want to call the better model or not.
For example,
When a student is writing an essay but misinterprets the prompt, you should output:
[["call": "YES", "You need to call the better model to alert the student that they are misunderstanding the prompt"]]
When a student is doing a math problem, but is able to work on it themself, you should output:
[["call": "NO", "Don't call the better model, because the student is able to work on it themselves, and doesn't need help"]]
When a student is on any digital classroom app, like Brightspace or Google Classroom, is looking at events online or on an app like Evite, is on an itinerary planning/flight/hotel app, like Expedia, Southwest, or other similar apps, or is looking at any other kind of thing that should be added to a calendar, you should output:
[["call": "YES", "The better model needs to add the event/itinerary/assignment to the user's calendar"]]
When a user is on any application/web browser that handles data (budgeting, finance, plotting graphs in Python, Excel/Google Sheets spreadsheet, or any other similar application), you should output:
[["call": "YES", "The better model needs to be called to create a graph/visualization/do calculations"]]
When a user has a need for any kind of graphics or interactive visualization, you should output:
[["call": "YES", "The better model needs to be called to create an interactive visualization"]]
When a user has any kind of code window or code in a text file open, or any other reason to write code, you should output:
[["call": "YES", "The better model needs to be called so it can ask the user whether they need help or not"]]
When a user is on a site where you are unsure if they need help or not, you should output:
[["call": "YES", "The better model needs to be called so it can ask the user whether they needs help or not"]]
The user can decide what to show the AI app, so if there is any sensitive or confidential information in the screenshot, it means the user is ok to show that information to the app. If any sensitive/confidential information shows up in the screenshot, but there are actionable items for the larger model, you should still call it.
"""

PROACTIVE_SYSTEM_PROMPT = """
You are a helpful and proactive AI assistant named Proto. You are observing the user's screen.
- Based on the provided screenshot, offer a concise, relevant, and actionable suggestion.
- You have access to tools: a code interpreter and Google Search. Use them when necessary to provide better answers, generate content, or get real-time information.
- When you use a tool, think step-by-step.
- Start your response directly with your observation or suggestion. Your entire response must be a single line of text.
"""

CHAT_SYSTEM_PROMPT = """
You are a helpful AI assistant named Proto. You are in a direct chat with the user.
- Answer their questions clearly and concisely.
- You have access to tools: a code interpreter for calculations and image generation, and Google Search for up-to-date information.
- Use your tools when a user's request requires them. For example, use code execution for "draw a chart of this data" or "calculate the fibonacci sequence". Use search for "who won the game last night?".
"""

class ChatMessage(BaseModel):
    message: str
    history: List[Dict]

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

async def stream_gemini_with_tools(model_name: str, contents: List[Dict], api_key: str, system_instruction: str) -> AsyncGenerator[Dict, None]:
    """
    Performs a streaming request to the Gemini API using aiohttp for native async performance.
    """
    request_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse"
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    payload = {
        "contents": contents,
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "tools": [{"google_search": {}}]
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(request_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logging.error(f"HTTP Error: {response.status} - {response.reason} - {error_text}")
                    yield {"type": "error", "content": f"API Error: {error_text}"}
                    return
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data: "):
                        try:
                            chunk = json.loads(line_str[6:])
                            if not chunk.get('candidates'): continue
                            
                            for part in chunk['candidates'][0].get('content', {}).get('parts', []):
                                if "text" in part and part["text"]:
                                    yield {'type': 'text', 'content': part['text']}
                                elif "toolCode" in part:
                                    yield {'type': 'search_start', 'content': 'Searching with Google...'}

                            if chunk['candidates'][0].get('groundingMetadata'):
                                metadata = chunk['candidates'][0]['groundingMetadata']
                                if metadata.get('webSearchQueries'):
                                    sources = [
                                        g_chunk.get('web', {}).get('uri') 
                                        for g_chunk in metadata.get('groundingChunks', []) 
                                        if g_chunk.get('web', {}).get('uri')
                                    ]
                                    if sources:
                                        yield {'type': 'search_end', 'content': list(set(sources))}
                        except json.JSONDecodeError:
                            logging.warning(f"Could not decode JSON chunk: {line_str[6:]}")
    except Exception as e:
        logging.error(f"Error in stream function: {e}", exc_info=True)
        yield {"type": "error", "content": f"An unexpected error occurred: {str(e)}"}

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.websocket("/ws/proactive")
async def websocket_proactive(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connection established for proactive AI.")
    try:
        while True:
            data_url = await websocket.receive_text()
            try:
                img_bytes = base64.b64decode(data_url.split(",")[1])
                img = Image.open(io.BytesIO(img_bytes))
            except Exception as e:
                logging.error(f"Could not process image data URL: {e}")
                continue
            
            triage_config = types.GenerateContentConfig(system_instruction=TRIAGE_SYSTEM_PROMPT)
            response = await client.aio.models.generate_content(
                model="models/gemini-2.5-flash-lite",
                contents=[img],
                config=triage_config
            )
            triage_response_full = response.text

            decision = None
            match = re.search(r'\[(\[.*?\])\]', triage_response_full, re.DOTALL)
            if match:
                try:
                    # vvvvvvvvvvvvvv FIX START vvvvvvvvvvvvvv
                    # The model sometimes returns [["key": "value"]] which is invalid JSON.
                    # This line corrects it to [{"key": "value"}] before parsing.
                    corrected_str = f"[{'{' + match.group(1).strip()[1:-1] + '}'}]"
                    # ^^^^^^^^^^^^^^^ FIX END ^^^^^^^^^^^^^^^
                    decision = json.loads(corrected_str)
                    logging.info(f"Triage Decision: {decision}")
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse triage JSON: {match.group(0)} - Error: {e}")
            
            if decision and decision[0].get("call") == "YES":
                logging.info("Triage is YES. Engaging Gemini 2.5 Pro with tools.")
                image_part_for_rest = {"inline_data": {"mime_type": "image/jpeg", "data": data_url.split(",")[1]}}
                proactive_contents = [{"role": "user", "parts": [image_part_for_rest, {"text": "Based on this screenshot, what is one helpful and proactive suggestion you can offer?"}]}]
                
                full_proactive_response = ""
                async for structured_chunk in stream_gemini_with_tools("gemini-2.5-pro", proactive_contents, API_KEY, PROACTIVE_SYSTEM_PROMPT):
                    if structured_chunk.get('type') == 'text':
                        full_proactive_response += structured_chunk['content']
                
                if full_proactive_response:
                    await websocket.send_json({"type": "proactive_suggestion", "content": full_proactive_response.strip()})

    except WebSocketDisconnect:
        logging.info("WebSocket connection closed.")
    except Exception as e:
        logging.error(f"Error in WebSocket: {e}", exc_info=True)
        try:
            await websocket.close(code=1011, reason="An internal error occurred.")
        except RuntimeError:
            pass 

async def chat_stream_generator(chat_message: ChatMessage):
    contents = chat_message.history + [{'role': 'user', 'parts': [{'text': chat_message.message}]}]
    async for structured_chunk in stream_gemini_with_tools("gemini-2.5-pro", contents, API_KEY, CHAT_SYSTEM_PROMPT):
        yield f"data: {json.dumps(structured_chunk)}\n\n"
        await asyncio.sleep(0.01)

@app.post("/chat")
async def chat_handler(chat_message: ChatMessage):
    return StreamingResponse(chat_stream_generator(chat_message), media_type="text/event-stream")