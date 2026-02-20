import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_ollama import ChatOllama

# Streamlit page config
st.set_page_config(page_title="PDF Summarizer", layout="wide")

st.title("📄 PDF Summarizer using Ollama")

# Sidebar for configuration
st.sidebar.header("Configuration")
# Model selection
model_name = st.sidebar.selectbox(
    "Ollama Model Name",
    ["llama3:8b", "tinyllama:latest"],
    index=0,
    help="Llama 3 is more capable but slower. TinyLlama is faster but less accurate."
)
st.sidebar.info("Make sure you have pulled the model using `ollama pull <model_name>`")

summary_mode = st.sidebar.radio(
    "Summarization Mode",
    ["Fast (Single Call)", "Detailed (Multi-Step)"],
    index=0,
    help="Fast: Sends whole text at once (good for short docs). Detailed: Splits doc and summarizes chunks (good for long docs)."
)

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    st.write(f"File uploaded successfully: {uploaded_file.name}")

    if st.button("Summarize PDF"):
        with st.spinner("Summarizing... This may take a while based on the document size."):
            try:
                # Initialize Ollama LLM
                llm = ChatOllama(model=model_name)
                
                # Load PDF
                loader = PyPDFLoader(temp_file_path)
                
                if summary_mode == "Fast (Single Call)":
                    # Load full document without splitting
                    docs = loader.load()
                    chain_type = "stuff"
                    st.info("Using 'Stuff' chain: Sending entire document context to LLM...")
                else:
                    # Detailed mode: Split with larger chunks to optimize speed
                    # Default is often small (e.g., 1000 or 500). Increasing to 3000 reduces # of LLM calls.
                    st.info("Using 'Map-Reduce' chain: Splitting document into optimized chunks...")
                    raw_docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
                    docs = text_splitter.split_documents(raw_docs)
                    chain_type = "map_reduce"
                    st.write(f"Processing {len(docs)} chunks...")

                # Check if we have documents
                if not docs:
                    st.error("No text could be extracted from the PDF.")
                else:
                    # Create summarization chain
                    chain = load_summarize_chain(llm, chain_type=chain_type)
                    
                    # Run the chain - use invoke instead of run
                    summary = chain.invoke(docs)
                    
                    st.subheader("Summary:")
                    # Handle different return types (some chains return dict, some str)
                    if isinstance(summary, dict) and 'output_text' in summary:
                        st.markdown(summary['output_text'])
                    else:
                        st.markdown(summary)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
