import streamlit as st
import tempfile

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------

st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("📘 Personal Knowledge Assistant (RAG System)")


# -------------------------------------------------------
# Load PDF
# -------------------------------------------------------

def load_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        path = tmp.name

    loader = PyPDFLoader(path)
    pages = loader.load()
    text = "\n".join([p.page_content for p in pages])
    return text


# -------------------------------------------------------
# Vectorstore
# -------------------------------------------------------

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


# -------------------------------------------------------
# RAG pipeline (New LangChain Style — works on Cloud)
# -------------------------------------------------------

def run_rag(vectordb, query):

    retriever = vectordb.as_retriever()

    llm = ChatGroq(
        groq_api_key=st.secrets["groq_api_key"],
        model="llama-3.1-8b-instant"
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Use ONLY the following context to answer:

    Context:
    {context}

    Question: {question}

    Answer:
    """)

    # 1. retrieve docs
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])

    # 2. generate answer
    chain_input = {"context": context, "question": query}
    response = llm.invoke(prompt.format(**chain_input))

    return response.content


# -------------------------------------------------------
# UI
# -------------------------------------------------------

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
