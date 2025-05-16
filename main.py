import os
import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Load PDF and process
loader = PyPDFLoader("docs/knowledge.pdf")
docs = loader.load()

# Split & embed
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(chunks, embedding)

# LLM & QA chain
retriever = vectordb.as_retriever()
llm = OpenAI(temperature=0)  # Gunakan OpenAI atau model lain sesuai pilihan
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Gradio interface
def chatbot(query):
    return qa_chain.run(query)

demo = gr.Interface(fn=chatbot, inputs="text", outputs="text", title="PDF Chatbot")

# Render-compatible launch
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
