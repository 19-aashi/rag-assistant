import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import tempfile

st.set_page_config(page_title="Personal Knowledge Assistant", layout="wide")
st.title("📘 Personal Knowledge Assistant (RAG System)")

def load_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    return "".join([p.page_content for p in pages])

@st.cache_resource
def create_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_texts(chunks, embedding=embeddings)
    return vectordb

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

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    st.success("PDF uploaded successfully!")

    text = load_pdf(uploaded)
    vectordb = create_vectorstore(text)

    query = st.text_input("Ask a question:")

    if query:
        with st.spinner("Generating answer..."):
            answer = run_rag(vectordb, query)
        st.write("### Answer:")
        st.write(answer)
