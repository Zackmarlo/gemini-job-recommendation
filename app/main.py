from fastapi import FastAPI , HTTPException , UploadFile , File
import requests
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from prompt_scheme import format_resume_prompt , parse_json , format_recommendation_prompt
from pdfloader import load_pdf

from sentence_transformers import SentenceTransformer, util
from recomendersys import preprocess_text

app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')


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


class JobDescription(BaseModel):
    title: str
    description: str

class JobRequest(BaseModel):
    user_skills: str
    job_descriptions: List[JobDescription]

@app.post("/recomendersystem")
def get_recommendations(data: JobRequest):
    # Preprocess user skills
    preprocessed_user_skills = preprocess_text(data.user_skills)
    user_embedding = model.encode(preprocessed_user_skills, convert_to_tensor=True)

    # Encode job descriptions
    job_embeddings = {
        job.title: model.encode(preprocess_text(job.description), convert_to_tensor=True)
        for job in data.job_descriptions
    }

    # Compute similarity scores
    results = {}
    for title, embedding in job_embeddings.items():
        similarity = util.cos_sim(user_embedding, embedding).item()
        results[title] = similarity

    # Sort recommendations by similarity score
    top_recommendations = sorted(results.items(), key=lambda x: x[1], reverse=True)

    # Return the top recommendations
    return {
        "recommendations": [
            {"job_title": job, "similarity_score": round(score, 4)}
            for job, score in top_recommendations
        ]
    }