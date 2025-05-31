# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot import Chatbot  # Your existing chatbot class
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from fastapi.responses import JSONResponse
import uvicorn

# ----------- FastAPI Setup -----------
app = FastAPI()

# Enable CORS (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------- Request Models -----------
class QueryRequest(BaseModel):
    question: str

class CareerRequest(BaseModel):
    message: dict

# ----------- Local JSON Chatbot Setup -----------
chatbot = Chatbot(json_path="data/jobs_data.json")

@app.post("/ask")
async def ask_question(query: QueryRequest):
    answer = chatbot.generate_answer(query.question)
    return {"answer": answer}

# ----------- Langchain Career Chat Setup -----------
groq_api = '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'  # Replace with your actual key
chat = ChatGroq(model='Gemma2-9b-It', groq_api_key=groq_api)

system_prompt = SystemMessage(content="""
You are a professional tech career guidance expert. Your task is to analyze a person's interests and relevant personal information (such as hobbies, personality traits, academic strengths, and goals) and recommend at least 5 career paths that align well with their profile. For each career path, provide a brief explanation of why it suits the individual, including how their interests and traits align with the demands and opportunities in that field. Be clear, practical, and insightful in your recommendations.
""")

@app.post("/chat")
async def chat_with_model(request: CareerRequest):
    try:
        userinfo_str = "\n".join(f"{k}: {v}" for k, v in request.message.items())
        user_prompt = f"{userinfo_str}\nSuggest careers with some details like experience, salary, category."

        messages = [
            system_prompt,
            HumanMessage(content=user_prompt)
        ]

        response = chat(messages)
        return JSONResponse(content={"response": response.content})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
