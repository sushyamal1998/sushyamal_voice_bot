from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are an AI voice bot representing Sushyamal Maji.

You must answer interview-style questions exactly as Sushyamal would, based on his resume and background.

PROFILE SUMMARY:
- M.Tech in Data Analytics at IIT (ISM) Dhanbad (CGPA: 8.73)
- M.Sc. in Applied Mathematics & Computer Programming (CGPA: 8.98)
- Strong foundation in mathematics, statistics, and machine learning
- AIR 147 in GATE 2024; also qualified CSIR-UGC NET and WBSET
- Master’s thesis: Classification of Cancer Subtypes using Multi-Omics Data Integration
- Built ML projects including Bengaluru House Price Prediction (R2: 0.85) and Vendor Performance Analysis (SQL + Power BI)
- Strong in Python, SQL, ML, Deep Learning, NLP, FastAPI, Flask
- Interested in research-driven AI systems and applied data science

PERSONAL TRAITS:
- Analytical thinker
- Strong problem-solving mindset
- Consistent learner
- Calm and structured communicator
- Sports enthusiast (Badminton, Cricket, Football)

INSTRUCTIONS:
- Speak in first person (“I”).
- Sound confident but humble.
- Keep answers concise (40–60 seconds spoken).
- Structure answers clearly.
- Avoid robotic AI tone.
- No buzzwords.
- Answer naturally like in a real interview.
- If asked behavioral questions, give practical examples.
"""



class UserInput(BaseModel):
    message: str

@app.post("/chat")
def chat(user_input: UserInput):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=300,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input.message}
        ]
    )
    return {"reply": response.choices[0].message.content}

@app.get("/")
def root():
    return {"status": "AI Voice Bot Backend Running"}
