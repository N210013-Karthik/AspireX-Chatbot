import os
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load and preprocess the data
def load_data(data_folder):
    questions, answers = [], []
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.json'):
            file_path = os.path.join(data_folder, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    json_data = json.load(file)
                    for item in json_data:
                        # Convert skills to readable string if it's a list
                        skills = item.get('skills_required', 'N/A')
                        if isinstance(skills, list):
                            skills = ', '.join(skills)

                        job_details = f"""
                        Job Name: {item.get('job_name', 'N/A')}
                        Category: {item.get('category', 'N/A')}
                        Description: {item.get('job_description', 'N/A')}
                        Eligibility: {item.get('eligibility', 'N/A')}
                        Skills Required: {skills}
                        Education Path: {item.get('education_path', 'N/A')}
                        Related exams: {item.get('related_exams', 'N/A')}
                        Career growth: {item.get('career_growth', 'N/A')}
                        Companies hiring: {item.get('companies_hiring', 'N/A')}
                        Salary: {item.get('salary_range', 'N/A')}
                        """
                        questions.append(job_details)
                        item['skills_required'] = skills  # Save updated skills as string
                        answers.append(item)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {file_name}")
    return questions, answers

# Step 2: Train the TF-IDF model
def train_model(questions):
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)
    return vectorizer, question_vectors

# Step 3: Qualification filter logic (Improved)
def filter_jobs_by_qualification(user_input, answers):
    qualification_keywords = ["graduation", "graduate", "bachelor", "bcom", "b.a.", "b.sc.", "mba"]
    # Extract predefined domains dynamically from the loaded data
    predefined_domains = list(set(job.get("category", "").lower() for job in answers if job.get("category")))

    # Find and extract domains from user input
    domain_keywords = [domain for domain in predefined_domains if domain in user_input.lower()]

    # Check if the user input contains any qualification keywords
    if any(k in user_input.lower() for k in qualification_keywords):
        matching_jobs = []
        for job in answers:
            # Check if the job matches both qualification and domain keywords
            if (
                any(k in job.get("eligibility", "").lower() for k in qualification_keywords) and
                any(d in job.get("category", "").lower() for d in domain_keywords)
            ):
                matching_jobs.append(job)
        return matching_jobs
    return []


# Step 4: Extract relevant fields for detailed answers
def extract_relevant_info(user_input, matched_answer):
    fields = {
        "job name": ("Job Name", matched_answer.get("job_name", "N/A")),
        "category": ("Category", matched_answer.get("category", "N/A")),
        "description": ("Description", matched_answer.get("job_description", "N/A")),
        "eligibility": ("Eligibility", matched_answer.get("eligibility", "N/A")),
        "skills": ("Skills Required", matched_answer.get("skills_required", "N/A")),
        "education": ("Education Path", matched_answer.get("education_path", "N/A")),
        "exams": ("Related Exams", matched_answer.get("related_exams", "N/A")),
        "growth": ("Career Growth", matched_answer.get("career_growth", "N/A")),
        "companies": ("Companies Hiring", matched_answer.get("companies_hiring", "N/A")),
        "salary": ("Salary Range", matched_answer.get("salary_range", "N/A")),
    }

    matched_fields = []
    for key, (label, value) in fields.items():
        if re.search(key, user_input, re.IGNORECASE):
            matched_fields.append(f"{label}: {value}")

    if matched_fields:
        return "Here’s what I found based on your question:\n" + "\n".join(matched_fields)

    summary = f"""
Sure! Here's a detailed summary of this job:
👉 **{matched_answer.get('job_name', 'N/A')}** (Category: {matched_answer.get('category', 'N/A')})
📌 Description: {matched_answer.get('job_description', 'N/A')}
🎯 Eligibility: {matched_answer.get('eligibility', 'N/A')}
🛠️ Skills Needed: {matched_answer.get('skills_required', 'N/A')}
🎓 Education Path: {matched_answer.get('education_path', 'N/A')}
📝 Related Exams: {matched_answer.get('related_exams', 'N/A')}
📈 Career Growth: {matched_answer.get('career_growth', 'N/A')}
🏢 Companies Hiring: {matched_answer.get('companies_hiring', 'N/A')}
💰 Salary Range: {matched_answer.get('salary_range', 'N/A')}
"""
    return summary.strip()


# Step 5: Chatbot logic
def chatbot(vectorizer, question_vectors, questions, answers):
    print("🎓 Career Guidance Chatbot is ready! Type your question or type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye and good luck with your career!")
            break

        # Check for graduation-type queries
        filtered_jobs = filter_jobs_by_qualification(user_input, answers)
        if filtered_jobs:
            response = "Here are some jobs you might be eligible for based on your qualification:\n"
            for job in filtered_jobs[:5]:
                response += f"• {job['job_name']} — {job['eligibility']}\n"
            print(f"Chatbot: {response.strip()}")
            continue

        # Fallback to semantic similarity
        user_vector = vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, question_vectors)
        best_match_idx = similarities.argmax()

        if similarities[0, best_match_idx] > 0.1:
            matched_answer = answers[best_match_idx]
            relevant_info = extract_relevant_info(user_input, matched_answer)
            print(f"Chatbot: {relevant_info}")
        else:
            print("Chatbot: Sorry, I couldn’t find any information related to your question.")

# Step 6: Main function
if __name__ == "__main__":
    data_folder = "./data"
    questions, answers = load_data(data_folder)
    if questions and answers:
        vectorizer, question_vectors = train_model(questions)
        chatbot(vectorizer, question_vectors, questions, answers)
    else:
        print("No job data found in the /data/ folder.")