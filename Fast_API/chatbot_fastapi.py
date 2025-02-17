import os
import torch
import shutil
import tempfile
import requests
import whisper
import gtts
from io import BytesIO
import threading
import time
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langdetect import detect
from pyngrok import ngrok
import nest_asyncio
import uvicorn
from dotenv import load_dotenv
load_dotenv()

nest_asyncio.apply()



llama_path = "chatbot_files/llama_finetuned"
faiss_path = "chatbot_files/faiss_index"

tokenizer = AutoTokenizer.from_pretrained(llama_path, token=os.getenv("HF_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(
    llama_path,
    torch_dtype=torch.float16,
    device_map="auto",
    token=os.getenv("HF_TOKEN")
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)

whisper_model = whisper.load_model("small")

app = FastAPI()

def retrieve_context(query, top_k=1):
    docs = vector_store.similarity_search(query, k=top_k)
    return "\n".join([doc.page_content for doc in docs])

import torch

def generate_response(query, detected_lang="en"):
    context = retrieve_context(query)

    final_prompt = f"""
    You are an expert in agriculture. Use the given context to answer accurately.

    Context:
    {context}

    Query:
    {query}

    Answer:
    """
    inputs = tokenizer(final_prompt, return_tensors="pt").to("cuda")
    output = model.generate(
        **inputs,
        max_length=256,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.split("Answer:")[-1].strip()

    if detected_lang != "en":
        translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{detected_lang}-en")
        response = translator(response)[0]["translation_text"]

    return response

class QueryRequest(BaseModel):
    text: str

@app.post("/chat/")
async def chat(request: QueryRequest):
    detected_lang = detect(request.text)
    response = generate_response(request.text, detected_lang)
    return JSONResponse({"response": response, "language_detected": detected_lang})

@app.post("/voice-chat/")
async def voice_chat(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
        shutil.copyfileobj(audio.file, temp_audio)
        temp_audio_path = temp_audio.name

    result = whisper_model.transcribe(temp_audio_path)
    text = result["text"]

    detected_lang = detect(text)
    response = generate_response(text, detected_lang)

    return JSONResponse({"transcribed_text": text, "response": response, "language_detected": detected_lang})

@app.post("/tts/")
async def tts(request: QueryRequest):
    detected_lang = detect(request.text)
    tts_audio = gtts.gTTS(request.text, lang=detected_lang)

    audio_io = BytesIO()
    tts_audio.write_to_fp(audio_io)
    audio_io.seek(0)

    return JSONResponse({"audio_url": "tts_audio.mp3", "language_detected": detected_lang})

# ngrok authtoken 2soUrM6W73EBC7cdACSCTwdwUbJ_3SKJtHJXYWtN1u2VgzuX6

def run_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=8000)

thread = threading.Thread(target=run_uvicorn)
thread.start()
time.sleep(5)

ngrok_tunnel = ngrok.connect(8000)
print(f"🚀 Public API URL: {ngrok_tunnel.public_url}")

