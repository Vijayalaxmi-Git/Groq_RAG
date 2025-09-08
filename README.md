# Groq_RAG

Groq_RAG is an AI application leveraging Retrieval-Augmented Generation (RAG) to provide rapid, contextually accurate responses by integrating:
Cohere: Generative Language Model
Groq: High-Performance Inference Engine
Streamlit: Interactive Web Interface
File/Folder	Purpose
app.py	Main application script; integrates Cohere for generation and Streamlit for UI.
requirements.txt	Lists necessary Python packages for the project.
Dockerfile	Defines the environment for containerization.
docker-compose.yaml	Configures multi-container Docker applications.
.gitignore	Specifies files and directories to be ignored by Git.
.gitpod.yml	Configures the Gitpod development environment.
LICENSE	Contains the project's licensing information.
README.md	Provides an overview and documentation for the project.
Flow diagram

User Query (Streamlit Input)
        │
        ▼
Vector Store / Index Search (Chroma + CohereEmbeddings)
        │
        ▼
Retrieve Top Relevant Documents
        │
        ▼
Cohere LLM Generates Answer (Using PromptTemplate / LLMChain)
        │
        ▼
Display Answer (Streamlit Output)
