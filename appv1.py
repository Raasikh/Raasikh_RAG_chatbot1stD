import os
import streamlit as st
import requests
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# 🔑 Load OpenAI API Key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# 🔗 Load FAISS vector index with HuggingFace embeddings
db = FAISS.load_local(
    "faiss_index",
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

# 🔍 Set retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# 🧠 Prompt Template
custom_prompt_template = """You are a helpful assistant that answers questions based only on the given context.

Use only the factual information in the context. If the answer is not contained within the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=["context", "question"]
)

# 🤖 Set OpenAI LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=openai_api_key
)

# 🔗 RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    input_key="question",  # 👈 This tells the chain to expect "question" input
    output_key="result"
)

# 🎨 Streamlit Frontend
st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
st.title("💬 LangChain + OpenAI RAG Chatbot")
query = st.text_input("Ask a question based on your documents:")

if query:
    with st.spinner("🤖 Thinking..."):
        result = qa_chain.invoke({"question": query})  # 👈 Match input key to template
        st.subheader("🧠 Answer:")
        st.write(result["result"])

        st.subheader("📚 Source Documents:")
        for doc in result["source_documents"]:
            st.markdown(f"• {doc.page_content}")

