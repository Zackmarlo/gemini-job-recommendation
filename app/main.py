from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional
from prompt_scheme import format_resume_prompt, parse_json, format_recommendation_prompt, hr_system_instructions
from pdfloader import load_pdf
from stream import stream_response
from sentence_transformers import SentenceTransformer, util
from recomendersys import preprocess_text
import whisper
import shutil
import os
import tempfile
import requests

# === FastAPI setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Global error handler for 422 ===
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    detailed_errors = [
        {
            "field": " -> ".join(str(i) for i in err["loc"]),
            "message": err["msg"],
            "type": err["type"]
        }
        for err in errors
    ]
    return JSONResponse(
        status_code=422,
        content={"detail": detailed_errors}
    )

# === Model loading ===
model_recommender = SentenceTransformer('all-MiniLM-L6-v2')
model_whisper = whisper.load_model("base", device="cpu")  # Use "cuda" if GPU available
ALLOWED_EXTENSIONS = [".mp3", ".wav"]

# === Gemini client setup ===
client = OpenAI(
    api_key="AIzaSyDu1kwjv0yFUkBi7U62JYS3PCLcsBfoiG8",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

chatbot_client = genai.Client(api_key="AIzaSyDu1kwjv0yFUkBi7U62JYS3PCLcsBfoiG8")


@app.get("/")
def root():
    return {"message": "Welcome to the custom API!"}


@app.post("/extract")
def parse_resume(resumefile: UploadFile = File(...)):
    try:
        contents = load_pdf(resumefile.file)
        prompt = format_resume_prompt(contents)

        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=prompt,
        )

        response_data = parse_json(response.choices[0].message.content)

        requests.post(
            "https://creative-endlessly-bullfrog.ngrok-free.app/api/ai/extract",
            json=response_data
        )

        return {"status": "Extraction successful", "data": response_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume extraction failed: {str(e)}")


# === Pydantic Models ===

class RecommendationRequest(BaseModel):
    skills: List[dict] = Field(..., min_items=1)
    work_experience: List[dict] = Field(..., min_items=1)
    projects: List[dict] = Field(..., min_items=1)

class JobDescription(BaseModel):
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=10)

class JobRequest(BaseModel):
    user_skills: str = Field(..., min_length=3)
    job_descriptions: List[JobDescription] = Field(..., min_items=1)

class ChatRequest(BaseModel):
    user_message: str = Field(..., min_length=1)
    chat_history: Optional[List[dict]] = None
    skills: List[dict]
    work_experience: List[dict]
    projects: List[dict]


@app.post("/recommend")
def recommend(info: RecommendationRequest):
    try:
        prompt = format_recommendation_prompt(info.skills, info.work_experience, info.projects)

        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=prompt,
        )

        top_recommendations = parse_json(response.choices[0].message.content)
        return {"response": top_recommendations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.post("/recomendersystem")
def get_recommendations(data: JobRequest):
    try:
        user_embedding = model_recommender.encode(preprocess_text(data.user_skills), convert_to_tensor=True)

        job_embeddings = {
            job.title: model_recommender.encode(preprocess_text(job.description), convert_to_tensor=True)
            for job in data.job_descriptions
        }

        results = {
            title: util.cos_sim(user_embedding, emb).item()
            for title, emb in job_embeddings.items()
        }

        top_recommendations = sorted(results.items(), key=lambda x: x[1], reverse=True)

        return {
            "recommendations": [
                {"job_title": job, "similarity_score": round(score, 4)}
                for job, score in top_recommendations
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding comparison failed: {str(e)}")


@app.post("/chat")
def chat(messageinfo: ChatRequest):
    try:
        chat = chatbot_client.chats.create(
            model="gemini-2.0-flash",
            history=messageinfo.chat_history or []
        )

        response = chat.send_message_stream(
            messageinfo.user_message,
            config=types.GenerateContentConfig(
                system_instruction=hr_system_instructions(
                    messageinfo.skills,
                    messageinfo.work_experience,
                    messageinfo.projects
                ),
            )
        )

        return StreamingResponse(
            stream_response(response),
            media_type="text/event-stream",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    ext = os.path.splitext(audio.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .mp3 and .wav files are allowed.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(audio.file, tmp)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save audio: {str(e)}")

    try:
        result = model_whisper.transcribe(tmp_path)
        os.remove(tmp_path)
        return {"transcription": result["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
