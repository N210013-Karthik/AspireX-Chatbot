# chatbot.py
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

class Chatbot:
    def __init__(self, json_path):
        self.api_key = os.getenv("GROQ_API_KEY")  # Set as environment variable
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.data = self._load_json(json_path)
        self.documents, self.doc_map = self._flatten_data(self.data)
        self.embeddings = self._generate_embeddings(self.documents)

    def _load_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _flatten_data(self, data):
        docs, mapping = [], []
        for category, items in data.items():
            for item in items:
                text = f"Category: {category}\n" + "\n".join(f"{k}: {v}" for k, v in item.items())
                docs.append(text)
                mapping.append(item)
        return docs, mapping

    def _generate_embeddings(self, docs):
        return self.model.encode(docs)

    def search(self, query, top_k=1):
        query_embedding = self.model.encode([query])
        scores = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = scores.argsort()[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

    def generate_answer(self, query):
        context = "\n\n".join(self.search(query, top_k=1))
        prompt = f"""You are a helpful assistant trained on career guidance data. Based on the context below, answer the user's question conversationally.

Context:
{context}

Question: {query}
Answer:"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
