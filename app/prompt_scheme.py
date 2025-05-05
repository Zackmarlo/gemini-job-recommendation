from pydantic import BaseModel, Field
from typing import List, Literal , Optional
import json
import json_repair

Month = Literal["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December" , None]
EducationLevel = Literal[
    "high_school", "undergraduate", "postgraduate", "doctorate", "not_specified"
]

JobLevel = Literal[
    "intern", "entry", "associate", "mid", "senior", "director", "executive", "not_specified"
]

Proficiency = Literal["beginner", "intermediate", "advanced", "expert"]

class PersonalInformation(BaseModel):
  name : str = Field(..., description="The name of the person")
  email : str = Field(..., description="The email of the person")
  phone : str = Field(..., description="The phone number of the person")
  address : str = Field(..., description="The address of the person")

class Skill(BaseModel):
  skill_name : str = Field(..., description="The name of the skill")
  proficiency_level : Proficiency = Field(..., description="The proficiency level of the skill")

class WorkExperience(BaseModel):
  job_title : str = Field(..., description="The title of the job")
  job_level : JobLevel = Field(..., description="The level of the job")
  company : str = Field(..., description="The company of the job")
  start_year : int = Field(..., description="the start year of the job")
  start_month : Month = Field(None, description="the start month of the job")
  end_year : int = Field(..., description="the end year of the job")
  end_month : Month = Field(None, description="the end month of the job")
  job_description : str = Field(..., description="The description of the job")

class Education(BaseModel):
  institution : str = Field(..., description="The institution of the education")
  degree : str = Field(..., description="The degree of the education")
  field_of_study : str = Field(..., description="The field of study of the education")
  start_year : int = Field(..., description="the start year of the job")
  start_month : Month = Field(None, description="the start month of the job")
  end_year : int = Field(..., description="the end year of the job")
  end_month : Month = Field(None, description="the end month of the job")
  education_level : EducationLevel = Field(..., description="The level of the education")

class Project(BaseModel):
  project_name : str = Field(..., description="The name of the project")
  start_date : str = Field(..., description="The start date of the project")
  end_date : str = Field(..., description="The end date of the project")
  URL : str = Field(..., description="The URL of the project")
  description : str = Field(..., description="The description of the project")

class FullInfo(BaseModel):
  personal_information : PersonalInformation = Field(..., description="The personal information of the person")
  skills : List[Skill] = Field(..., description="The skills of the person")
  work_experiences : List[WorkExperience] = Field(..., description="The work experiences of the person")
  educations : List[Education] = Field(..., description="The educations of the person")
  projects : List[Project] = Field(..., description="The projects of the person")

def format_resume_prompt(resume):
  data_extraction_prompt = [
      {
          "role": "system",
          "content": "\n".join([
              "You are an NLP data paraser.",
              "You will be provided by a resume text associated with a Pydantic scheme.",
              "You have to extract JSON details from text according the Pydantic details.",
              "Extract details as mentioned in text.",
              "Do not generate any introduction or conclusion.",
              "if you did not find the data in the text return None"

          ])
      },
      {
          "role": "user",
          "content": "\n".join([
              "## resume",
              resume.strip(),

              "",
              "## Pydantic Details:",
              json.dumps(
                  FullInfo.model_json_schema(),
                  ensure_ascii=False
                  ),

              "",

              "## Details:",
              "```json"
          ])
      }
  ]
  return data_extraction_prompt

def parse_json(data):
  try:
    return json_repair.loads(data)
  except:
    return None
  
class Strength(BaseModel):
  skill_name : str = Field(..., description="The name of the skill")
  proficiency_level : Proficiency = Field(..., description="The proficiency level of the skill")
  reason: Optional[str] = Field(None, description="Why this skill is considered a strength")

class Weakness(BaseModel):
  skill_name : str = Field(..., description="A skill that is either missing from the candidate's resume or present but below the job's required level")
  proficiency_level : Proficiency = Field(..., description="The candidate's current level of this skill. Use 'None' if the skill is completely missing.")
  required_level: Optional[Proficiency] = Field(None, description="The expected level for this job")
  improvement_tips: Optional[str] = Field(None, description="Suggestions to improve or learn this skill")

class Job(BaseModel):
  job_title : str = Field(..., description="The title of the job")
  job_level : JobLevel = Field(..., description="The level of the job (e.g., Entry, Mid, Senior)")
  strengths_points : List[Strength] = Field(..., description="The skills the candidate already has that match or exceed the job requirements")
  weakness_points: List[Weakness] = Field(..., description="The skills that are either missing entirely or present at a lower-than-required proficiency")

class JobRecommendation(BaseModel):
  jobs : List[Job] = Field(..., description="The recommended jobs for the customer")
  reason: Optional[str] = Field(None, description="Why this job is recommended")

def format_recommendation_prompt(skills, work_experience, projects):
  job_recommendation_prompt = [
      {
          "role":"system",
          "content": "\n".join([
              "You are an NLP job recommender.",
              "You will be provided by some skills, projects and work experience associated with a Pydantic scheme.",
              "You have to extract JSON details from text according the Pydantic details.",
              "Recommend the top 5 matched job based on the data you get",
              "Extract details as mentioned in text.",
              "Do not generate any introduction or conclusion."
          ])
      },
      {
          "role": "user",
          "content": "\n".join([
              "## skills",
              json.dumps(skills, ensure_ascii=False),

              "",
              "## projects",
              json.dumps(projects, ensure_ascii=False),

              "",
              "## work_experience",
              json.dumps(work_experience, ensure_ascii=False),

              "",
              "## Pydantic Details:",
              json.dumps(
                  JobRecommendation.model_json_schema(),
                  ensure_ascii=False
                  ),

              "",

              "## Details:",
              "```json"
          ])
      }
  ]
  return job_recommendation_prompt

def hr_system_instructions(skills, work_experience, projects):
  system_message = "\n".join([
              "You are a professional HR assistant conducting mock interviews.",
              "Start by asking the user: 'What job role would you like to be interviewed for?'",
              "After the user responds, begin a structured interview for that role.",
              "Ask questions relevant to their provided skills, experience, and projects.",
              "Probe technical knowledge based on skills and project context.",
              "Evaluate communication clarity and depth of answers.",
              "Avoid repeating questions and keep the tone supportive but professional.",
              "mock the the company interview process that suits the user's skills and experience and level",
              "keep it short and concise",
              "After the interview, provide feedback on strengths and areas for improvement.",
              f"skills: {skills}",
              f"work_experience: {work_experience}",  
              f"projects: {projects}"
          ])
  return system_message