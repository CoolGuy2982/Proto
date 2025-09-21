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
from fastapi.responses import HTMLResponse
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
# --- System Prompts for Gemini ---
TRIAGE_SYSTEM_PROMPT = """
You are a triage model. Your job is to observe a user's screen activity over a short period and decide if a more powerful AI assistant should intervene.

You will receive a sequence of screenshots and your own previous verdicts. The sequence is chronological, with the last image being the most recent screen.

Based on the entire sequence, you must decide if the user's *current* action (the latest screenshot) warrants help.

Look for patterns:
- Is the user struggling with something over multiple screens (e.g., debugging code, re-writing an essay prompt)?
- Has a new, actionable item appeared (e.g., an assignment, an event, a data table)?
- Is the user navigating to a site where they might need help (e.g., coding editor, calendar, spreadsheet app)?

Respond ONLY in the specified JSON format: [["call": "YES", "reason"]] or [["call": "NO", "reason"]].

Your reason should be concise and explain your decision based on the context.

Example scenarios:
- If you see a user looking at a calendar, then navigating to a new event invitation, you should call the model:
[["call": "YES", "The user is looking at an event invitation that should be added to their calendar."]]
- If you see a user typing in a code editor and making progress, you don't need to intervene:
[["call": "NO", "The user is actively coding and doesn't appear to be stuck."]]
- If you see a user switch from a code editor with an error to a Stack Overflow page, they might need help:
[["call": "YES", "The better model needs to be called so it can ask the user whether they need help or not"]]
"""

PROACTIVE_SYSTEM_PROMPT = """
You are a proactive AI assistant. You will be shown a chronological sequence of a user's screen activity, which may include screenshots and the verdicts from a smaller AI model that decided to call you.

Your task is to analyze this entire sequence to understand the user's context and goal. Based on this understanding, provide a single, highly relevant, and helpful suggestion related to the user's *most recent* activity (the last screenshot).

For example, if you see the user first looking at their class schedule, and then looking at a specific assignment page, your suggestion should be about adding that specific assignment to their calendar, mentioning the class context.

After analyzing the context, follow these specific instructions for your output:

// for looking at user screenshot and generating calendar events from the items on the image
FOR THE OUTPUT 
1. you shouldn't be generating a code execution, you should be adding the event into google calendar by formatting the customizable create event link with the information you got from the screenshotted image from the user:
To create a link for adding an event to Google Calendar, use the following format: 
[["calendar":"https://calendar.google.com/calendar/render?action=TEMPLATE&text=[EVENT_TITLE]&dates=[START_DATE_TIME]/[END_DATE_TIME]&details=[EVENT_DETAILS]&location=[EVENT_LOCATION]"]]. 
You must URL-encode the values for title, details, and location, and format dates and times correctly. For timed events, use the format  YYYYMMDDTHHMMSS with a ctz (time zone) parameter for the local time of the user 

2. if you see multiple events or multiple assignments etc, ask if the user wants to add all of them to their calendar, then add the events

3. If the event does not a location specified and you cannot find anything on google search that clues you into where the event is, put the best general location, e.g. for a class homework due for CS 193 and you can see that it's on the Purdue University WL brightspace, if you can't find where the building that this class is being held, just put the location as a general location like Purdue University.

4. I would not like you to ask me for confirmation to create/add event(s), i would just like you to create the events and i can confirm them by clicking the link or not

// for looking at user screenshot and generating data visualizations accordingly
You're going to get an image input, and you need to use another AI to output and display some sort of visualization or graph based on that input. You should pretend like you're creating the data visualization when you're actually asking another AI to do it.
If you see multiple tables of data, generate graphs/visualizations for all of them.
Generate the original visualization based on how you think the data is best displayed, but if the user says they want another type of visualization, output that in place of your original one.
Provide your output in the same format as this example:
[["demo":"Imagine a line graph here with 'Day' on the x-axis (1, 2, 3, 4) and 'Temperature' on the y-axis, showing points at (1, 20), (2, 22), (3, 19), and (4, 25), connected by a line."]]


// this is for coding help or proactively asking if the user needs guidance/help
1. when you get an image input from the smaller AI model but haven't gotten a specified reason why, you should prompt the user with whether they need help or not. 
2. you should read the screenshotted image input from the user's screen to determine what they could possibly need help with, and ask if they need help with a specific area.
3. for example, if you can see code on the image, ask if they need help finishing writing the code that they are working on by analyzing what the code they've written so far could possibly do, ask to specify what their final goal is with their code, and provide them with tips/coding help on whichever path they specify
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
            # Receive history and current image from the client
            json_data = await websocket.receive_text()

            # If the received message is empty, ignore it and wait for the next one.
            if not json_data:
                continue

            data = json.loads(json_data)
            history = data.get('history', [])
            current_image_data_url = data.get('current_image')

            if not current_image_data_url:
                continue

            # --- Triage Model Call ---
            # Construct the multi-modal history for the triage model
            triage_contents = []
            for item in history:
                try:
                    hist_img_bytes = base64.b64decode(item['image'].split(",")[1])
                    hist_img = Image.open(io.BytesIO(hist_img_bytes))
                    triage_contents.append(hist_img)
                    #triage_contents.append(item['verdict'])
                except Exception as e:
                    logging.warning(f"Skipping malformed history item: {e}")
                    continue
            
            # Add the current image to the contents for triage
            current_img_bytes = base64.b64decode(current_image_data_url.split(",")[1])
            current_img = Image.open(io.BytesIO(current_img_bytes))
            triage_contents.append(current_img)
            
            triage_config = types.GenerateContentConfig(response_mime_type="application/json", 
                                                        system_instruction=TRIAGE_SYSTEM_PROMPT)
            response = await client.aio.models.generate_content(
                model="models/gemini-2.5-flash-lite",
                contents=triage_contents,
                config=triage_config,
            )
            triage_response_full = response.text
            
            # Initialize the response payload for the frontend
            response_payload = {
                "type": "proactive_update",
                "verdict": triage_response_full,
                "suggestion": None
            }

            # --- Proactive Model Call (if triage is YES) ---
            decision = None
            match = re.search(r'\[(\[.*?\])\]', triage_response_full, re.DOTALL)
            if match:
                try:
                    corrected_str = f"[{'{' + match.group(1).strip()[1:-1] + '}'}]"
                    decision = json.loads(corrected_str)
                    logging.info(f"Triage Decision: {decision}")
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse triage JSON: {match.group(0)} - Error: {e}")
            
            if decision and decision[0].get("call") == "YES":
                logging.info("Triage is YES. Engaging Gemini 2.5 Pro with tools.")
                
                # Construct the multi-modal history for the proactive model (REST API format)
                proactive_contents = []
                for item in history:
                    proactive_contents.append({
                        "role": "user",
                        "parts": [{"inline_data": {"mime_type": "image/jpeg", "data": item['image'].split(",")[1]}}]
                    })
                    proactive_contents.append({
                        "role": "model",
                        "parts": [{"text": item['verdict']}]
                    })
                
                # Add current image and the final prompt
                proactive_contents.append({
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": current_image_data_url.split(",")[1]}},
                        {"text": "Based on the sequence of screenshots showing my recent activity, what is one helpful and proactive suggestion you can offer?"}
                    ]
                })

                full_proactive_response = ""
                sources = []
                async for structured_chunk in stream_gemini_with_tools("gemini-2.5-flash", proactive_contents, API_KEY, PROACTIVE_SYSTEM_PROMPT):
                    chunk_type = structured_chunk.get('type')
                    if chunk_type == 'text':
                        full_proactive_response += structured_chunk['content']
                    elif chunk_type == 'search_end':
                        sources.extend(structured_chunk.get('content', []))
                
                if full_proactive_response:
                    response_payload["suggestion"] = {
                        "content": full_proactive_response.strip(),
                        "sources": list(set(sources))
                    }

            # Send the consolidated update to the client
            await websocket.send_json(response_payload)

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



HTML_SYSTEM_PROMPT = """
You're going to get an input of a description of a graph and you need to create an HTML file to build an appropriate graph for the data.
Just output the HTML file, don't include any background information or explanations.
The HTML file must use dimensions that fit perfectly in a square. EVERYTHING MUST FIT IN A 1:1 CONTAINER!
Make it interactive if needed. For example, the user should be able to change the values of initial velocity and angle in a projectile motion simulation.
Use the chart.js JavaScript library for data visualizations and similar graphs, with the CDN link https://cdn.jsdelivr.net/npm/chart.js
For animations, games, and other interactive visualizations, use the p5.js JavaScript library with the CDN link https://cdn.jsdelivr.net/npm/p5@1.11.10/lib/p5.min.js
"""


# This can be the same client instance you use elsewhere
htmlClient = client # or genai.Client() if you want a separate one

async def html_generator(chat_message: ChatMessage) -> str:
    response = await htmlClient.aio.models.generate_content(
        model="models/gemini-2.5-pro",
        contents=[{"role": "user", "parts": [{"text": chat_message.message}]}],
        config=types.GenerateContentConfig(
            system_instruction=HTML_SYSTEM_PROMPT,
            response_mime_type="text/plain"
        )
    )
    text = response.text
    match = re.search(r'```(?:\w*\n)?(.*)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


@app.post("/html_gen")
async def html_handler(chat_message: ChatMessage):
    html_code = await html_generator(chat_message)
    return HTMLResponse(content=html_code)