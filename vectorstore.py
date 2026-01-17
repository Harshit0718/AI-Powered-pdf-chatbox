from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
import os

os.makedirs("vectorstore", exist_ok=True)

# Load your PDF
loader = PyPDFLoader("sample.pdf")  # put your PDF here
documents = loader.load()

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS DB
db = FAISS.from_documents(documents, embeddings)

# Save locally
db.save_local("vectorstore")

print("Vectorstore created successfully")
