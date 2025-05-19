import os
import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Persistent DB folder
CHROMA_DB_DIR = "chroma_db"

def setup_vectordb():
    if os.path.exists(CHROMA_DB_DIR):
        # Load prebuilt vectordb to avoid recomputing
        return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

    loader = PyPDFLoader("docs/knowledge.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # smaller chunk size for memory
    chunks = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding, persist_directory=CHROMA_DB_DIR)
    vectordb.persist()
    return vectordb

# Set up only once
vectordb = setup_vectordb()
retriever = vectordb.as_retriever()
llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def chatbot(query):
    return qa_chain.run(query)

demo = gr.Interface(fn=chatbot, inputs="text", outputs="text", title="PDF Chatbot")

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
