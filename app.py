import os
import json
import logging
import asyncio
import re  # <--- IMPORT THE REGEX MODULE
from typing import List, Dict, AsyncGenerator

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- Configuration and Setup ---
logging.basicConfig(level=logging.INFO)
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it.")

# --- System Prompts for Gemini ---

# vvvvvvvvvvvvvv CHANGE START vvvvvvvvvvvvvv
TRIAGE_SYSTEM_PROMPT = """
You are a highly efficient triage AI. Your only task is to analyze a screenshot and decide if a proactive AI assistant's help would be useful.
- Look for signs of user struggle, complex tasks, coding, design work, or writing.
- Do NOT offer help for simple browsing, watching videos, or idle screens.
- Your response MUST be a single, valid JSON array containing one object.
- The format MUST be exactly: `[{"call": "YES"}]` or `[{"call": "NO"}]`.
- Do NOT include any other text, explanations, or markdown formatting like ```json. Your entire output must be only the JSON itself.
"""
# ^^^^^^^^^^^^^^^ CHANGE END ^^^^^^^^^^^^^^^

PROACTIVE_SYSTEM_PROMPT = """
You are a helpful and proactive AI assistant named Proto. You are observing the user's screen.
- Based on the provided screenshot, offer a concise, relevant, and actionable suggestion.
- If it's code, provide a code snippet.
- If it's design, suggest an improvement or a component.
- If it's writing, suggest a rephrasing or a next step.
- Keep your suggestion brief and to the point. Start your response directly with your observation or suggestion.
- You have a code execution tool. Use it to generate images or other artifacts if the user's context implies a visual task (e.g., 'create a button', 'design a logo').
"""

CHAT_SYSTEM_PROMPT = """
You are a helpful AI assistant named Proto. You are in a direct chat with the user.
- Answer their questions clearly and concisely.
- You have a code execution tool available. Use it when the user asks for something that requires it, like generating an image or running a calculation.
"""

class ChatMessage(BaseModel):
    message: str
    history: List[Dict]

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

async def stream_gemini_manually(model_name: str, contents: List[Dict], api_key: str, system_instruction: str) -> AsyncGenerator[Dict, None]:
    request_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    payload = {
        "contents": contents,
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "tools": [{"code_execution": {}}]
    }
    async with aiohttp.ClientSession(read_bufsize=25 * 1024 * 1024) as session:
        try:
            async with session.post(request_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data: "):
                        try:
                            yield json.loads(line_str[6:])
                        except json.JSONDecodeError:
                            logging.warning(f"Could not decode JSON chunk: {line_str[6:]}")
        except aiohttp.ClientResponseError as e:
            logging.error(f"HTTP Error during manual stream: {e.status} - {e.message} - Headers: {e.headers}")
            yield {"error": f"HTTP {e.status}: {e.message}"}
        except Exception as e:
            logging.error(f"Unexpected error during manual Gemini stream: {e}")
            yield {"error": f"An unexpected error occurred: {str(e)}"}

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
            image_part = {"inline_data": {"mime_type": "image/jpeg", "data": data_url.split(",")[1]}}
            triage_contents = [{"role": "user", "parts": [image_part]}]
            
            triage_response_full = ""
            async for chunk in stream_gemini_manually("gemini-2.5-flash-lite", triage_contents, API_KEY, TRIAGE_SYSTEM_PROMPT):
                if chunk.get('candidates'):
                    triage_response_full += chunk['candidates'][0]['content']['parts'][0]['text']
            
            # vvvvvvvvvvvvvv CHANGE START vvvvvvvvvvvvvv
            # Robust JSON parsing using regex to extract the JSON part
            decision = None
            # This regex finds the first occurrence of a JSON array (`[...]`) in the string
            match = re.search(r'(\[.*?\])', triage_response_full)

            if match:
                json_string = match.group(1)
                try:
                    decision = json.loads(json_string)
                    logging.info(f"Successfully parsed Triage JSON: {decision}")
                except json.JSONDecodeError:
                    logging.warning(f"Regex found a JSON-like string, but it failed to parse: {json_string}")
            else:
                logging.warning(f"Could not find any JSON in the triage response: {triage_response_full}")

            # Now, check the parsed decision
            if decision and isinstance(decision, list) and len(decision) > 0 and decision[0].get("call") == "YES":
                logging.info("Triage is YES. Engaging Gemini 2.5 Pro.")
                proactive_contents = [{"role": "user", "parts": [image_part]}]
                async for chunk in stream_gemini_manually("gemini-2.5-pro", proactive_contents, API_KEY, PROACTIVE_SYSTEM_PROMPT):
                    await websocket.send_json(chunk)
            # ^^^^^^^^^^^^^^^ CHANGE END ^^^^^^^^^^^^^^^

    except WebSocketDisconnect:
        logging.info("WebSocket connection closed.")
    except Exception as e:
        logging.error(f"Error in WebSocket: {e}", exc_info=True)
        await websocket.close(code=1011, reason="An internal error occurred.")


async def chat_stream_generator(chat_message: ChatMessage):
    contents = chat_message.history + [{"role": "user", "parts": [{"text": chat_message.message}]}]
    async for chunk in stream_gemini_manually("gemini-2.5-pro", contents, API_KEY, CHAT_SYSTEM_PROMPT):
        if "error" in chunk:
            yield f"data: {json.dumps(chunk)}\n\n"
            break
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.01)

@app.post("/chat")
async def chat_handler(chat_message: ChatMessage):
    return StreamingResponse(chat_stream_generator(chat_message), media_type="text/event-stream")