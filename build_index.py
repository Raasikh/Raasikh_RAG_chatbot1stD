import os
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 📂 Load .docx documents using UnstructuredFileLoader
def load_docx_files(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            file_path = os.path.join(folder_path, filename)
            loader = UnstructuredFileLoader(file_path)
            docs.extend(loader.load())
    return docs

# 📁 Define path to your macOS Downloads folder
downloads_path = os.path.expanduser("~/Downloads/your_docs")
docs = load_docx_files(downloads_path)

# ✂️ Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# 🧠 Create embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 📚 Create FAISS vector store
db = FAISS.from_documents(split_docs, embedding_model)

# 💾 Save FAISS index
db.save_local("faiss_index")

print("✅ FAISS index created and saved from DOCX files.")

