import sys
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.agents import initialize_agent, Tool

# ------------------ LOAD API KEYS ------------------
load_dotenv()
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ------------------ MODEL SETUP ------------------
# Text embedding model (Cohere)
embedding_model = CohereEmbeddings(model="embed-english-light-v3.0")

# GROQ model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ------------------ PDF & VECTORSTORE HELPERS ------------------
def load_pdf_and_split(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

def get_vectorstore(splits):
    vectorstore = Chroma.from_documents(
        splits, embedding=embedding_model, persist_directory="./chroma_db"
    )

# ------------------ RAG & AGENT SETUP ------------------
# Simple calculator tool
def calculator_fn(query: str):
    try:
        return str(eval(query))
    except Exception as e:
        return f"Error in calculation: {e}"

tools = [
    Tool(
        name="Calculator",
        func=calculator_fn,
        description="Use this tool to solve math problems"
    ),
]

# Function to get RAG chain using your existing llm and embeddings
def get_rag_chain():
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    retriever = db.as_retriever()
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

# Agent helper function
def query_agent(user_input: str):
    # If input looks like math, use calculator tool
    if any(op in user_input for op in ["+", "-", "*", "/"]):
        agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)
        return agent.run(user_input)
    else:
        rag_chain = get_rag_chain()
        return rag_chain.invoke({"input": user_input})["answer"]

# ------------------ STREAMLIT UI ------------------
if __name__ == "__main__":
    st.set_page_config(page_title="RAG Chatbot with GROQ + Cohere Agent", layout="wide")
    st.title("RAG Chatbot with GROQ + Cohere Agent")
    st.subheader("Upload PDFs and get embeddings first")

    # User query input
    query = st.text_input("Ask a question about the uploaded document")
    submit = st.button("Answer")

    # PDF uploader in sidebar
    with st.sidebar:
        uploaded_file = st.file_uploader("Please Upload PDF", type=["pdf"])
        if uploaded_file:
            temp_file = "./temp.pdf"
            with open(temp_file, "wb") as file:
                file.write(uploaded_file.getvalue())
            if st.button("Get Embeddings"):
                with st.spinner("Processing..."):
                    splits = load_pdf_and_split(temp_file)
                    get_vectorstore(splits)
                    st.success("Embeddings created successfully!")

    # Handle user query with agent
    if submit and query:
        with st.spinner("Processing your query..."):
            response = query_agent(query)
        st.write(response)
