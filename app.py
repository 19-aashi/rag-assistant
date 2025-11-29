import streamlit as st
import tempfile

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("📘 Personal Knowledge Assistant (RAG System)")


# -------------------------------
# PDF Loader
# -------------------------------

def load_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        path = tmp.name

    loader = PyPDFLoader(path)
    pages = loader.load()
    text = "\n".join([p.page_content for p in pages])
    return text


# -------------------------------
# Vectorstore Creation
# -------------------------------

@st.cache_resource
def create_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_texts(chunks, embedding=embeddings)
    return vectordb


# -------------------------------
# RAG Pipeline
# -------------------------------

def run_rag(vectordb, query):
    llm = ChatGroq(
        groq_api_key=st.secrets["groq_api_key"],
        model="llama-3.1-8b-instant"
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever()
    )

    return qa.run(query)


# -------------------------------
# UI Controls
# -------------------------------

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    st.success("PDF uploaded successfully!")

    with st.spinner("Processing PDF... (first time takes longer)"):
        text = load_pdf(uploaded)
        vectordb = create_vectorstore(text)

    st.success("Vectorstore ready! Ask anything from your document.")

    query = st.text_input("Ask a question:")
    if query:
        with st.spinner("Thinking..."):
            answer = run_rag(vectordb, query)

        st.subheader("Answer:")
        st.write(answer)
