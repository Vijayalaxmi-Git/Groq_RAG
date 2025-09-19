# Groq_RAG

Groq_RAG is an AI-powered chatbot application that leverages **Retrieval-Augmented Generation (RAG)** using Cohere and Groq models to provide fast, contextually accurate responses to user queries over uploaded PDF documents.

## Features

- **PDF Ingestion**: Upload PDF files via a web interface.
- **Document Chunking**: Split large documents into manageable text chunks for better semantic indexing.
- **Semantic Search**: Uses Cohere embeddings and Chroma vector store for fast, relevant document retrieval.
- **Retrieval-Augmented Generation (RAG)**: Combines retrieved context chunks with Groq LLM to generate accurate answers.
- **Agent AI**: Dynamically selects specialized tools (e.g., calculator for math queries) or routes queries to RAG pipeline.
- **Interactive Web Interface**: Built with Streamlit for easy user interaction.
- **Containerized Deployment**: Includes Docker support for seamless environment setup.

---

### In Detail

1. **PDF Upload**: Users upload PDFs through Streamlit sidebar.
2. **Document Processing**: PDFs are loaded and split into chunks using LangChain's `RecursiveCharacterTextSplitter`.
3. **Embedding Generation**: Each chunk is embedded using Cohere's embedding model and stored in a local Chroma DB.
4. **Query Handling**:
    - If the query is a math expression, LangChain's agent AI invokes a Python calculator tool.
    - Otherwise, the query is semantically matched to relevant chunks via Chroma, and Groq LLM generates the final response using these context chunks.
5. **Answer Presentation**: The answer is streamed back to the user via the Streamlit web interface.

---

## Tech Stack

- **Python**: Main programming language.
- **Streamlit**: Interactive web app framework.
- **LangChain**: Orchestration of LLMs, embeddings, and agents.
- **Chroma**: Vector database for storing document embeddings.
- **Cohere**: Embedding and generative language models.
- **Groq**: High-performance inference engine for LLMs.
- **Docker**: Containerization for reproducible deployments.
- **dotenv**: Secure management of API keys.

## How It Works

### 1. PDF Upload & Embedding

- PDFs are uploaded and saved temporarily.
- The file is loaded using `PyPDFLoader` and split into chunks.
- Each chunk is embedded via Cohere and stored in Chroma DB.

### 2. Query Handling

- **Math Queries**: If user input is a mathematical expression (e.g., `12*5+3`), the agent AI uses a calculator tool (`eval()` in Python) for precise calculation.
- **General Queries**: For other queries, the system retrieves relevant document chunks and uses Groq LLM to generate answers with context-awareness.

### 3. Streamlit UI

- Users interact via a clean UI:
    - PDF upload (sidebar)
    - Query input (main pane)
    - Answer display (main pane)

---

## Installation

### Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)
- API keys for Cohere and Groq

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Vijayalaxmi-Git/Groq_RAG.git
   cd Groq_RAG
   ```

2. **Install Dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Set API Keys**

   Create a `.env` file in the root with:

   ```
   COHERE_API_KEY=your_cohere_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Run the Application**

   ```bash
   streamlit run app.py
   ```

5. **Or use Docker**

   ```bash
   docker build -t groq_rag .
   docker run -p 8501:8501 groq_rag
   ```

---

## Usage

- Open the app in your browser (`localhost:8501`).
- Upload a PDF using the sidebar.
- Click "Get Embeddings" to process and index the document.
- Enter your query (about the document or a math expression) in the main input.
- Click "Answer" to get results.

---

## Customization

- **Add More Tools**: Extend the agent AI by adding new tools (e.g., summarization, translation, search).
- **Change Models**: Swap out Groq/Cohere models for others supported by LangChain.
- **UI Enhancements**: Customize Streamlit components for richer user experience.

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [LangChain](https://python.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [Cohere](https://cohere.com/)
- [Groq](https://groq.com/)
- [Chroma](https://www.trychroma.com/)
