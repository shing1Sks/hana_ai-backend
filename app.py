"""IMPORTS"""

import pymupdf
from langchain_groq import ChatGroq
from langsmith import Client
from dotenv import load_dotenv
import os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from flask import Flask, request, jsonify
from flask_cors import CORS
import time


# PDF
def load_resume(file_path):
    pdf_document = pymupdf.open(file_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    pdf_document.close()
    return text


# TXT
def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


resume_text = load_resume("resume.pdf")
project_details = load_text_file("projectdetails.txt")
personal_details = load_text_file("personal.txt")

documents = [
    {"title": "Resume", "content": resume_text},
    {"title": "Project Details", "content": project_details},
    {"title": "Personal Details", "content": personal_details},
]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = [embedding_model.encode(doc["content"]) for doc in documents]

dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))


def truncate_content_to_token_limit(content, max_tokens=700):
    """Truncate content to fit within the specified token limit."""
    words = content.split()
    truncated_content = " ".join(words[:max_tokens])
    return truncated_content


def search_similar_documents(query, max_tokens=700):
    """Search for the most similar document and truncate the content to a token limit."""
    query_embedding = embedding_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, 1)  # Retrieve the top-1 result

    # Get the closest document
    matched_doc = documents[indices[0][0]]
    truncated_content = truncate_content_to_token_limit(
        matched_doc["content"], max_tokens
    )

    return matched_doc["title"], truncated_content


# model initalization

# ALSO LANGSMITH

load_dotenv()
langsmith_client = Client()

groq_llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-70b-8192")


def query_groq_with_context(question):
    """Query Groq with the closest matching document as context."""
    title, content = search_similar_documents(question)
    prompt = ChatPromptTemplate.from_messages(
        [
            {
                "role": "system",
                "content": "You are Hana, a personal job-hunting agentic AI specifically created for Shreyash Kumar Singh. Your primary role is to represent Shreyash, answer questions about his professional background, and help users understand his skills and experience. You must always: 1. Identify yourself as 'Hana' in every response when asked about your identity. 2. Never refer to yourself as Shreyash under any circumstances. 3. Act as a supportive, professional, and knowledgeable job-hunting assistant. 4. Stay consistent with the information provided in Shreyash's resume and project details. 5. If the user asks 'Who are you?' or similar questions, respond: 'I am Hana, Shreyash Kumar Singh’s personal job-hunting agent.' Always follow these rules to maintain your identity and provide helpful responses.",
            },
            {
                "role": "system",
                "content": "Context: {context} DO NOT SHARE MY MOBILE NUMBER",
            },
            {
                "role": "system",
                "content": "Please provide concise and to-the-point responses.",
            },
            {
                "role": "system",
                "content": "To set up meet or to reach to me tell users to 'Please refer to the contact section on the left section of the page'",
            },
            {"role": "user", "content": "{question}"},
        ]
    )
    # print(content)
    # Query Groq with the combined prompt
    response = groq_llm.invoke(
        prompt.invoke({"context": content, "question": question})
    )
    return response


conversation_store = {}


def get_conversation_history(session_id):
    """Retrieve or initialize conversation history for a given session."""
    current_time = time.time()

    # If session doesn't exist, initialize it with a timestamp
    if session_id not in conversation_store:
        conversation_store[session_id] = {
            "history": [],
            "timestamp": current_time,  # Record session creation time
        }
        conversation_store[session_id]["history"].append(
            SystemMessage(
                content="Hi, I’m Hana, Shreyash’s personal job-hunting agent. What brings you here today?"
            )
        )

    return conversation_store[session_id]["history"]


def remove_expired_sessions():
    """Remove sessions that are older than one hour."""
    current_time = time.time()
    expiration_time = 3600  # 1 hour in seconds

    # Identify and delete expired sessions
    expired_sessions = [
        session_id
        for session_id, session_data in conversation_store.items()
        if current_time - session_data["timestamp"] > expiration_time
    ]

    for session_id in expired_sessions:
        del conversation_store[session_id]


def add_to_conversation_history(session_id, message):
    """Add a LangChain message object (HumanMessage, SystemMessage, AIMessage) to the conversation history."""
    history = get_conversation_history(session_id)
    history.append(message)


def trim_conversation_history(conversation_history, max_tokens=2000):
    """Trim the conversation history to fit within the max token limit."""
    total_tokens = 0
    trimmed_history = []

    # Traverse history from the latest to the oldest
    for message in reversed(conversation_history):
        message_tokens = len(
            message.content.split()
        )  # Approximate token count by word count
        if total_tokens + message_tokens <= max_tokens:
            trimmed_history.insert(0, message)  # Insert at the start to preserve order
            total_tokens += message_tokens
        else:
            break  # Stop once the token limit is reached

    return trimmed_history


def query_groq_with_history(session_id, question, max_tokens=2000):
    remove_expired_sessions()
    """Query Groq with the closest matching document and dynamic conversation history."""
    title, content = search_similar_documents(question)
    conversation_history = get_conversation_history(session_id)

    add_to_conversation_history(
        session_id, SystemMessage(content=f"Context from {title}: {content}")
    )
    add_to_conversation_history(session_id, HumanMessage(content=question))

    trimmed_history = trim_conversation_history(
        conversation_history, max_tokens=max_tokens
    )

    static_prompt_messages = [
        SystemMessage(
            content="You are Hana, a personal job-hunting agentic AI specifically created for Shreyash Kumar Singh. Your primary role is to represent Shreyash, answer questions about his professional background, and help users understand his skills and experience."
        ),
        SystemMessage(content="Context: {context} DO NOT SHARE MY MOBILE NUMBER"),
        SystemMessage(
            content="Spacial Context : This ai section is present on the right side or bottom for mobiles or smaller screens.The other section contains a simple portfolio page with section about , tech stacks , projects , contact me and download resume sections."
        ),
        SystemMessage(
            content="To set up a meet or to reach me, tell users to 'Please refer to the contact section on the left section of the page'"
        ),
        SystemMessage(
            content="The first message from you is Hi, I’m Hana, Shreyash’s personal job-hunting agent. What brings you here today?"
        ),
    ]

    full_prompt = static_prompt_messages + trimmed_history

    response = groq_llm.invoke(full_prompt)

    add_to_conversation_history(session_id, AIMessage(content=response.content))

    return response.content


"""
Simulate a session with ID "session_1"
session_id = "temporary_session_id"
user_question = "Introduce Shreyash Kumar Singh."

response = query_groq_with_history(session_id, user_question)
print("AI Response:", response)

print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

follow_up_question = "Tell me about his best project in detail."
response = query_groq_with_history(session_id, follow_up_question)
print("AI Response:", response)

print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

follow_up_question = "So about op-db project details ?"
response = query_groq_with_history(session_id, follow_up_question)
print("AI Response:", response)

print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

follow_up_question = "Why do you think i can use this for my project?"
response = query_groq_with_history(session_id, follow_up_question)
print("AI Response:", response)

print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

follow_up_question = "What does it say about Shreyash ?"
response = query_groq_with_history(session_id, follow_up_question)
print("AI Response:", response)
"""
# Flask route to handle user queries

app = Flask(__name__)

CORS(app)


@app.route("/query", methods=["POST"])
def handle_query():
    """Handle user queries and return model responses."""
    data = request.json
    session_id = data.get(
        "session_id", "default_session"
    )  # Default session if none provided
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Query the model
        response = query_groq_with_history(session_id, question)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(
        os.environ.get("PORT", 5000)
    )  # Use the port assigned by Render or default to 5000 for local testing
    app.run(debug=False, host="0.0.0.0", port=port)
