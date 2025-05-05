from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.metrics.pairwise import cosine_similarity
from chatbot import load_data, train_model, filter_jobs_by_qualification, extract_relevant_info

# Initialize FastAPI app and Jinja2 template engine
app = FastAPI()
templates = Jinja2Templates(directory="static")

# Load data and train model at startup
data_folder = "./data"
questions, answers = load_data(data_folder)
vectorizer, question_vectors = train_model(questions)

class Query(BaseModel):
    question: str

def format_job_list(jobs: list) -> str:
    """Format a list of job dicts into a readable bullet list."""
    return "\n".join([f"• {job['job_name']} — {job.get('eligibility', 'Eligibility not specified')}" for job in jobs])

def get_bot_response(question: str) -> str:
    """Core logic to handle the chatbot response based on question."""
    # Step 1: Try to find jobs based on qualification
    filtered_jobs = filter_jobs_by_qualification(question, answers)
    if filtered_jobs:
        return "Jobs available for you:\n" + format_job_list(filtered_jobs)

    # Step 2: Fallback to semantic similarity
    user_vector = vectorizer.transform([question])
    similarities = cosine_similarity(user_vector, question_vectors)
    matching_indices = (similarities > 0.1).nonzero()[1]

    if matching_indices.size > 0:
        matched_answers = [answers[idx] for idx in matching_indices]

        # Check if the query matches an exact job name
        exact_match = next((answer for answer in matched_answers if isinstance(answer, dict) and answer.get("job_name", "").lower() == question.lower()), None)
        if exact_match:
            details = "\n".join([f"{key.capitalize()}: {value}" for key, value in exact_match.items()])
            return f"Exact match found:\n{details}"

        # If no exact match, format and return all matched jobs
        if matched_answers and all(isinstance(answer, dict) for answer in matched_answers):
            jobs = [{"job_name": answer["job_name"], "eligibility": answer["eligibility"]} for answer in matched_answers]
            return "Jobs related to your query:\n" + format_job_list(jobs)

        # Else extract and return relevant info from all matches
        return "\n\n".join([extract_relevant_info(question, answer) for answer in matched_answers])

    return "Sorry, I couldn’t find relevant information."

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render homepage with input form."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_class=HTMLResponse)
async def ask_bot_website(request: Request, question: str = Form(...)):
    """Handle form submission from website and return response."""
    response = get_bot_response(question)
    return templates.TemplateResponse("index.html", {"request": request, "response": response, "question": question})

@app.post("/ask/api", response_class=JSONResponse)
async def ask_bot_api(question: str = Form(...)):
    """API endpoint to return chatbot response as JSON."""
    response = get_bot_response(question)
    return JSONResponse(content={"response": response})
