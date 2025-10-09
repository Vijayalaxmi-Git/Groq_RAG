import streamlit as st
import os
import hashlib
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Tuple
import re
import numexpr as ne

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = Path("./cache")
CHROMA_DIR = Path("./chroma_db")
METADATA_FILE = CACHE_DIR / "metadata.json"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_RETRIEVAL_DOCS = 4

# Create necessary directories
CACHE_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# Initialize session state
if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="result"
    )
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}
if 'total_tokens' not in st.session_state:
    st.session_state.total_tokens = 0
if 'api_calls' not in st.session_state:
    st.session_state.api_calls = {'cohere': 0, 'groq': 0}


class SafeCalculator:
    """Safe calculator that doesn't use eval()"""
    
    @staticmethod
    def is_math_expression(text: str) -> bool:
        """Check if input is a mathematical expression"""
        # Allow numbers, operators, parentheses, and spaces only
        pattern = r'^[\d\s\+\-\*/\(\)\.\^%]+$'
        return bool(re.match(pattern, text.strip()))
    
    @staticmethod
    def calculate(expression: str) -> str:
        """Safely evaluate mathematical expression using numexpr"""
        try:
            expression = expression.strip()
            
            # Security check
            if not SafeCalculator.is_math_expression(expression):
                return "Invalid expression. Only numbers and basic operators allowed."
            
            # Replace ^ with ** for exponentiation
            expression = expression.replace('^', '**')
            
            # Use numexpr for safe evaluation
            result = ne.evaluate(expression)
            logger.info(f"Math calculation: {expression} = {result}")
            
            return f"The result is: {result}"
            
        except Exception as e:
            logger.error(f"Math calculation error: {str(e)}")
            return f"Error in calculation: {str(e)}"


class DocumentCache:
    """Manages document caching to avoid reprocessing"""
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Generate MD5 hash of file content"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def load_metadata() -> Dict:
        """Load cache metadata"""
        if METADATA_FILE.exists():
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    @staticmethod
    def save_metadata(metadata: Dict):
        """Save cache metadata"""
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @staticmethod
    def is_cached(file_hash: str) -> bool:
        """Check if document is already processed"""
        metadata = DocumentCache.load_metadata()
        return file_hash in metadata
    
    @staticmethod
    def add_to_cache(file_hash: str, filename: str, num_chunks: int):
        """Add processed document to cache"""
        metadata = DocumentCache.load_metadata()
        metadata[file_hash] = {
            'filename': filename,
            'num_chunks': num_chunks,
            'processed_at': datetime.now().isoformat(),
        }
        DocumentCache.save_metadata(metadata)
        logger.info(f"Cached document: {filename} (hash: {file_hash})")


class CostTracker:
    """Track API usage and estimated costs"""
    
    # Approximate costs (update with current pricing)
    COSTS = {
        'cohere_embed': 0.0001,  # per 1K tokens
        'groq_llm': 0.0001,      # per 1K tokens (varies by model)
    }
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    @staticmethod
    def track_embedding(text_length: int):
        """Track embedding API call"""
        tokens = text_length // 4
        st.session_state.total_tokens += tokens
        st.session_state.api_calls['cohere'] += 1
        logger.info(f"Embedding tokens: {tokens}")
    
    @staticmethod
    def track_llm(prompt: str, response: str):
        """Track LLM API call"""
        tokens = CostTracker.estimate_tokens(prompt + response)
        st.session_state.total_tokens += tokens
        st.session_state.api_calls['groq'] += 1
        logger.info(f"LLM tokens: {tokens}")
    
    @staticmethod
    def get_estimated_cost() -> float:
        """Calculate estimated cost"""
        embed_cost = (st.session_state.api_calls['cohere'] * 1000 * CostTracker.COSTS['cohere_embed']) / 1000
        llm_cost = (st.session_state.total_tokens * CostTracker.COSTS['groq_llm']) / 1000
        return embed_cost + llm_cost


def process_pdf(pdf_path: str, file_hash: str) -> Chroma:
    """Process PDF and create vector store with caching"""
    
    # Check cache first
    if DocumentCache.is_cached(file_hash):
        logger.info(f"Loading from cache: {file_hash}")
        st.info("✅ Document already processed! Loading from cache...")
        
        embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=os.getenv("COHERE_API_KEY")
        )
        
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR / file_hash),
            embedding_function=embeddings
        )
        return vectorstore
    
    # Process new document
    st.info("🔄 Processing new document...")
    progress_bar = st.progress(0)
    
    try:
        # Load PDF
        progress_bar.progress(20)
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        # Defensive: check if pages is empty
        if not pages or len(pages) == 0:
            logger.error("No pages found in PDF. Is the document empty or corrupted?")
            st.error("No pages found in PDF. Please upload a valid PDF document.")
            return None
    
        logger.info(f"After Loaded {len(pages)} pages from PDF")        
        st.write(f"Loaded pages: {pages}")
        
    
    
        # Split into chunks
        progress_bar.progress(40)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(pages)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['source'] = os.path.basename(pdf_path)
        
        # Create embeddings
        progress_bar.progress(60)
        embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=os.getenv("COHERE_API_KEY")
        )
        
        # Track cost
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        CostTracker.track_embedding(total_chars)
        
        # Create and persist vector store
        progress_bar.progress(80)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_DIR / file_hash)
        )
        
        # Save to cache
        DocumentCache.add_to_cache(file_hash, os.path.basename(pdf_path), len(chunks))
        
        progress_bar.progress(100)
        st.success(f"✅ Processed {len(chunks)} chunks successfully!")
        
        return vectorstore
        
    except Exception as e:
        if "Stream has ended unexpectedly" in str(e):
            st.error("Vijaya The PDF file is corrupted or incomplete. Please upload a valid PDF.")
        else:
            logger.error(f"Error processing PDF: {str(e)}")
            st.error(f"Error processing PDF: {str(e)}")
        raise


def create_optimized_prompt() -> PromptTemplate:
    """Create an optimized prompt template for better responses"""
    
    template = """You are a helpful AI assistant analyzing documents. Your goal is to provide accurate, concise answers based solely on the provided context.

Context from the document:
{context}

Conversation History:
{chat_history}

Question: {question}

Instructions:
1. Answer ONLY based on the context provided above
2. If the context doesn't contain the answer, say "I cannot find this information in the provided document"
3. Be specific and cite page numbers when possible (look for 'page' in metadata)
4. Keep answers concise but complete
5. If relevant, reference the conversation history for context

Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "question"]
    )


def create_rag_chain(vectorstore: Chroma) -> RetrievalQA:
    """Create optimized RAG chain with custom prompt"""
    
    # Initialize LLM
    llm = ChatGroq(
        temperature=0.3,  # Lower temperature for more focused answers
        model_name="llama-3.1-70b-versatile",  # Use appropriate model
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Create retriever with optimization
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": MAX_RETRIEVAL_DOCS,  # Number of chunks to retrieve
        }
    )
    
    # Create chain with custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Stuffs all context into one prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": create_optimized_prompt(),
            "memory": st.session_state.conversation_memory
        }
    )
    
    return qa_chain


def create_agent(qa_chain):
    """Create agent with safe calculator and document QA tools"""
    
    # Safe calculator tool
    calculator = Tool(
        name="Calculator",
        func=SafeCalculator.calculate,
        description="Useful for mathematical calculations. Input should be a math expression like '2+2' or '(10*5)/2'. Only use this for pure math questions."
    )
    
    # Document QA tool
    def qa_wrapper(query: str) -> str:
        try:
            result = qa_chain({"query": query})
            answer = result['result']
            
            # Track cost
            CostTracker.track_llm(query, answer)
            
            # Add source information
            sources = result.get('source_documents', [])
            if sources:
                pages = set([doc.metadata.get('page', 'Unknown') for doc in sources])
                answer += f"\n\n📄 Sources: Pages {', '.join(map(str, sorted(pages)))}"
            
            return answer
        except Exception as e:
            logger.error(f"QA error: {str(e)}")
            return f"Error processing query: {str(e)}"
    
    document_qa = Tool(
        name="Document_QA",
        func=qa_wrapper,
        description="Use this to answer questions about the uploaded PDF document. Input should be a natural language question."
    )
    
    # Initialize LLM for agent
    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.1-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Create agent
    agent = initialize_agent(
        tools=[calculator, document_qa],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=st.session_state.conversation_memory,
        handle_parsing_errors=True,
        max_iterations=3  # Prevent infinite loops
    )
    
    return agent


def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Enhanced Groq RAG",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Enhanced Groq RAG Chatbot")
    st.caption("Secure, Cached, and Cost-Optimized Document Q&A")
    
    # Sidebar
    with st.sidebar:
        st.header("📄 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=['pdf'],
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file:
            # Save uploaded file temporarily
            temp_path = CACHE_DIR / uploaded_file.name
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Calculate hash
            file_hash = DocumentCache.get_file_hash(str(temp_path))
            
            # Display cache status
            if DocumentCache.is_cached(file_hash):
                st.success("✅ Document in cache")
            else:
                st.info("🆕 New document")
            
            # Process button
            if st.button("🚀 Process Document", type="primary"):
                try:
                    vectorstore = process_pdf(str(temp_path), file_hash)
                    
                    # Create RAG chain and agent
                    qa_chain = create_rag_chain(vectorstore)
                    agent = create_agent(qa_chain)
                    
                    # Store in session state
                    st.session_state.agent = agent
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.processed_files[file_hash] = uploaded_file.name
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Cost tracking
        st.divider()
        st.header("💰 Usage Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cohere Calls", st.session_state.api_calls['cohere'])
        with col2:
            st.metric("Groq Calls", st.session_state.api_calls['groq'])
        
        st.metric("Estimated Cost", f"${CostTracker.get_estimated_cost():.4f}")
        st.metric("Total Tokens", f"{st.session_state.total_tokens:,}")
        
        # Clear cache button
        if st.button("🗑️ Clear Cache"):
            st.session_state.conversation_memory.clear()
            st.session_state.total_tokens = 0
            st.session_state.api_calls = {'cohere': 0, 'groq': 0}
            st.success("Cache cleared!")
            st.rerun()
    
    # Main chat interface
    if 'agent' in st.session_state:
        st.success(f"📖 Analyzing: {st.session_state.current_file}")
        
        # Query input
        query = st.text_input(
            "Ask a question about your document or perform calculations:",
            placeholder="e.g., 'What are the key findings?' or '25 * 48 + 100'",
            key="query_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            ask_button = st.button("💬 Ask", type="primary")
        with col2:
            if st.button("🔄 New Conversation"):
                st.session_state.conversation_memory.clear()
                st.success("Conversation reset!")
                st.rerun()
        
        if ask_button and query:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Check if it's a math query
                    if SafeCalculator.is_math_expression(query):
                        st.info("🔢 Detected mathematical expression")
                    
                    # Get response from agent
                    response = st.session_state.agent.run(query)
                    
                    # Display response
                    st.markdown("### 💡 Answer:")
                    st.write(response)
                    
                    # Debug info (optional)
                    with st.expander("🔍 Debug Information"):
                        st.write(f"**Query:** {query}")
                        st.write(f"**Response Length:** {len(response)} characters")
                        st.write(f"**Estimated Tokens:** {CostTracker.estimate_tokens(query + response)}")
                    
                except Exception as e:
                    logger.error(f"Query error: {str(e)}")
                    st.error(f"Error processing query: {str(e)}")
        
        # Conversation history
        if st.session_state.conversation_memory.chat_memory.messages:
            with st.expander("📜 Conversation History"):
                for msg in st.session_state.conversation_memory.chat_memory.messages:
                    role = "You" if msg.type == "human" else "Assistant"
                    st.write(f"**{role}:** {msg.content}")
    
    else:
        st.info("👈 Upload a PDF document to get started!")
        
        # Example queries
        st.markdown("### 💡 Example Queries")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Document Questions:**
            - What is the main topic of this document?
            - Summarize section 3
            - What does page 5 say about...?
            """)
        with col2:
            st.markdown("""
            **Math Calculations:**
            - 25 * 48 + 100
            - (500 - 150) / 5
            - 2^10
            """)


if __name__ == "__main__":
    main()
