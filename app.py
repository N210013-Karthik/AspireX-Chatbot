from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from sklearn.metrics.pairwise import cosine_similarity
from chatbot import load_data, train_model, filter_jobs_by_qualification, extract_relevant_info

# FastAPI setup
app = FastAPI()
templates = Jinja2Templates(directory="static")

# Load and train the model
data_folder = "./data"
questions, answers = load_data(data_folder)
vectorizer, question_vectors = train_model(questions)

class Query(BaseModel):
    question: str

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_bot(request: Request, question: str = Form(...)):
    # Check for jobs available based on graduation
    filtered_jobs = filter_jobs_by_qualification(question, answers)
    
    if filtered_jobs:
        jobs = [f"• {j['job_name']} — {j['eligibility']}" for j in filtered_jobs[:5]]
        response = "Jobs available for you based on your graduation:\n" + "\n".join(jobs)
        return templates.TemplateResponse("index.html", {"request": request, "response": response, "question": question})

    # Fallback to semantic similarity if no qualification-based jobs found
    user_vector = vectorizer.transform([question])
    similarities = cosine_similarity(user_vector, question_vectors)
    best_match_idx = similarities.argmax()

    if similarities[0, best_match_idx] > 0.1:
        matched_answer = answers[best_match_idx]
        relevant_info = extract_relevant_info(question, matched_answer)
        return templates.TemplateResponse("index.html", {"request": request, "response": relevant_info, "question": question})
    else:
        response = "Sorry, I couldn’t find relevant information."
        return templates.TemplateResponse("index.html", {"request": request, "response": response, "question": question})

