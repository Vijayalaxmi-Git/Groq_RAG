import sys

# Fix for ChromaDB sqlite3 version issue
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
from dotenv import load_dotenv
import os
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load api keys
load_dotenv()
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Text embedding model
embedding_model = CohereEmbeddings(model="embed-english-light-v3.0")

# GROQ model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


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


# Get rag response
def get_response(retriever, query):
    # Answer question
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
    response = rag_chain.invoke({"input": query})
    return response["answer"]


if __name__ == "__main__":
    st.set_page_config(page_title="RAG-Utkarsh", layout="wide")
    st.title("RAG Chatbot - Utkarsh")
    st.subheader("Please upload and get embeddings first")
    query = st.text_input("Ask Question about the uploaded document")
    submit = st.button("Answer")
    with st.sidebar:
        uploaded_file = st.file_uploader("Please Upload PDF", type=["pdf"])
        if uploaded_file:
            temp_file = "./temp.pdf"
            with open(temp_file, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            if st.button("Get Embeddings"):
                with st.spinner("processing ..."):
                    splits = load_pdf_and_split(temp_file)
                    get_vectorstore(splits)
                    st.success("Done")
    if submit:
        with st.spinner("responding..."):
            db = Chroma(
                persist_directory="./chroma_db", embedding_function=embedding_model
            )
            retriever = db.as_retriever()
            response = get_response(retriever, query)
            st.write(response)
