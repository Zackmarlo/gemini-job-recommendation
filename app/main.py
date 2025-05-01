from fastapi import FastAPI , HTTPException , UploadFile , File
import requests
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from prompt_scheme import format_resume_prompt , parse_json , format_recommendation_prompt
from pdfloader import load_pdf

app = FastAPI()

client = OpenAI(
    api_key= "AIzaSyDu1kwjv0yFUkBi7U62JYS3PCLcsBfoiG8",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

@app.get("/")
def root():
    return {"message": "Welcome to the custom API!"}

@app.post("/extract")
def parse_resume(resumefile: UploadFile = File(...)):
    # Read the contents of the uploaded file
    contents = load_pdf(resumefile.file)
    prompt = format_resume_prompt(contents)

    response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=prompt,
        )
    response_data = parse_json(response.choices[0].message.content)
    return response_data

class RecommendationRequest(BaseModel):
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
        raise HTTPException(status_code=500, detail=str(e))
