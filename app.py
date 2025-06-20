import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from langchain.prompts import PromptTemplate
import requests
from typing import List

# 🧠 Custom LLM class for LM Studio using /v1/completions
class LMStudioLLM(BaseLLM):
    model: str = "mistral-7b-instruct-v0.3"
    temperature: float = 0.7
    endpoint: str = "http://localhost:1234/v1/completions"

    def _generate(self, prompts: List[str], stop=None, run_manager=None, **kwargs) -> LLMResult:
        generations = []
        for prompt in prompts:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": 512
            }
            try:
                response = requests.post(self.endpoint, json=payload)
                response.raise_for_status()
                content = response.json()["choices"][0]["text"]
                generations.append([Generation(text=content.strip())])
            except Exception as e:
                generations.append([Generation(text=f"❌ Error from LM Studio: {str(e)}")])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self):
        return "lm_studio"

# 📥 Load FAISS index
db = FAISS.load_local(
    "faiss_index",
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

# 🔍 Create retriever
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

# 🔗 QA Chain
llm = LMStudioLLM()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# 🎨 Streamlit UI
st.title("💬 LangChain + LM Studio (Mistral) RAG Chatbot")
query = st.text_input("Ask a question based on your documents:")

if query:
    with st.spinner("🤖 Thinking..."):
        result = qa_chain.invoke({"query": query})
        st.subheader("🧠 Answer:")
        st.write(result["result"])

        st.subheader("📚 Source Documents:")
        for doc in result["source_documents"]:
            st.markdown(f"• {doc.page_content}")
