"""
Enhanced Groq_RAG - Debug Version
Run: streamlit run app.py
"""

import streamlit as st

# Test 1: Basic Streamlit
st.title("🤖 Enhanced Groq RAG - Debug Mode")
st.write("✅ Streamlit is working!")

# Test 2: Check imports
st.subheader("Checking Dependencies...")

try:
    import os
    st.success("✅ os")
except Exception as e:
    st.error(f"❌ os: {e}")

try:
    import hashlib
    st.success("✅ hashlib")
except Exception as e:
    st.error(f"❌ hashlib: {e}")

try:
    import json
    st.success("✅ json")
except Exception as e:
    st.error(f"❌ json: {e}")

try:
    from pathlib import Path
    st.success("✅ pathlib")
except Exception as e:
    st.error(f"❌ pathlib: {e}")

try:
    import logging
    st.success("✅ logging")
except Exception as e:
    st.error(f"❌ logging: {e}")

try:
    import re
    st.success("✅ re")
except Exception as e:
    st.error(f"❌ re: {e}")

try:
    from dotenv import load_dotenv
    st.success("✅ dotenv")
except Exception as e:
    st.error(f"❌ dotenv: {e}")
    st.warning("Install: pip install python-dotenv")

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    st.success("✅ langchain.text_splitter")
except Exception as e:
    st.error(f"❌ langchain.text_splitter: {e}")
    st.warning("Install: pip install langchain")

try:
    from langchain_community.document_loaders import PyPDFLoader
    st.success("✅ langchain_community")
except Exception as e:
    st.error(f"❌ langchain_community: {e}")
    st.warning("Install: pip install langchain-community pypdf")

try:
    from langchain_community.vectorstores import Chroma
    st.success("✅ Chroma")
except Exception as e:
    st.error(f"❌ Chroma: {e}")
    st.warning("Install: pip install chromadb")

try:
    from langchain_cohere import CohereEmbeddings
    st.success("✅ langchain_cohere")
except Exception as e:
    st.error(f"❌ langchain_cohere: {e}")
    st.warning("Install: pip install langchain-cohere cohere")

try:
    from langchain_groq import ChatGroq
    st.success("✅ langchain_groq")
except Exception as e:
    st.error(f"❌ langchain_groq: {e}")
    st.warning("Install: pip install langchain-groq")

try:
    import numexpr as ne
    st.success("✅ numexpr")
    # Test it
    result = ne.evaluate("2+2")
    st.info(f"   Test: 2+2 = {result}")
except Exception as e:
    st.error(f"❌ numexpr: {e}")
    st.warning("Install: pip install numexpr")

# Test 3: Check .env file
st.subheader("Checking Environment Variables...")

try:
    from dotenv import load_dotenv
    load_dotenv()
    
    cohere_key = os.getenv("COHERE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if cohere_key and cohere_key != "your_cohere_api_key_here":
        st.success(f"✅ COHERE_API_KEY found (length: {len(cohere_key)})")
    else:
        st.error("❌ COHERE_API_KEY not set or using placeholder")
        st.warning("Add to .env file: COHERE_API_KEY=your_actual_key")
    
    if groq_key and groq_key != "your_groq_api_key_here":
        st.success(f"✅ GROQ_API_KEY found (length: {len(groq_key)})")
    else:
        st.error("❌ GROQ_API_KEY not set or using placeholder")
        st.warning("Add to .env file: GROQ_API_KEY=your_actual_key")
        
except Exception as e:
    st.error(f"Error loading .env: {e}")

# Test 4: Check directory structure
st.subheader("Checking Directories...")

required_dirs = ['cache', 'chroma_db']
for dir_name in required_dirs:
    if Path(dir_name).exists():
        st.success(f"✅ {dir_name}/ exists")
    else:
        st.warning(f"⚠️ {dir_name}/ missing - will be created")
        try:
            Path(dir_name).mkdir(exist_ok=True)
            st.success(f"   Created {dir_name}/")
        except Exception as e:
            st.error(f"   Failed to create {dir_name}/: {e}")

# Test 5: Check if .env exists
st.subheader("Checking Files...")

if Path(".env").exists():
    st.success("✅ .env file exists")
    with st.expander("View .env content (masked)"):
        try:
            with open(".env", "r") as f:
                content = f.read()
                # Mask API keys
                masked = re.sub(r'(API_KEY=)(.+)', r'\1***MASKED***', content)
                st.code(masked)
        except Exception as e:
            st.error(f"Error reading .env: {e}")
else:
    st.error("❌ .env file not found")
    st.warning("Create .env file with:")
    st.code("""COHERE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here""")
    
    if st.button("Create .env template"):
        with open(".env", "w") as f:
            f.write("COHERE_API_KEY=your_cohere_api_key_here\n")
            f.write("GROQ_API_KEY=your_groq_api_key_here\n")
        st.success("Created .env template - please add your API keys!")
        st.rerun()

# Summary
st.divider()
st.subheader("📊 Summary")

st.info("""
**If all checks pass:**
1. Replace this file with the full app.py
2. Run: `streamlit run app.py`

**If checks fail:**
1. Install missing packages (see warnings above)
2. Create/update .env file with your API keys
3. Re-run this debug script
""")

st.divider()
st.caption("Debug mode - Replace with full app.py once all checks pass")
