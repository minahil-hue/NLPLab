import os
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# --------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------
st.set_page_config(page_title="Hostel Chatbot", page_icon="🏠")

# --------------------------------------------------
# Hugging Face Token (from Streamlit Secrets)
# --------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")

# --------------------------------------------------
# Hostel Knowledge Base
# --------------------------------------------------
HOSTEL_INFO = """
HOSTEL INFORMATION

Room Types:
- Single Room: Rs. 5000/month, AC, WiFi, meals included
- Double Room: Rs. 3500/month per person, shared bathroom, WiFi
- Triple Room: Rs. 2800/month per person, common bathroom, WiFi
- Dormitory: Rs. 2000/month, shared space for 6-8 people

Facilities:
- 24/7 WiFi and hot water
- Laundry service twice a week
- Common kitchen (vegetarian only)
- Study room open 24/7
- Gym (6 AM to 9 PM)
- TV room with streaming

Rules:
- Entry time: 6 AM to 10 PM
- Visitors in common areas only (10 AM - 7 PM)
- No smoking or alcohol
- Keep noise low after 10 PM

Meal Timings:
- Breakfast: 7:30 AM - 9:30 AM
- Lunch: 12:30 PM - 2:30 PM
- Dinner: 7:30 PM - 9:30 PM

Booking Process:
1. Visit hostel or contact us
2. Fill registration form with ID proof
3. Pay security deposit: Rs. 5000 (refundable)
4. Pay first month rent
5. Get room keys and ID card
"""

# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# --------------------------------------------------
# Load Embeddings (Cached)
# --------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# --------------------------------------------------
# Load LLM (Cached)
# --------------------------------------------------
@st.cache_resource
def load_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-large",
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

# --------------------------------------------------
# Initialize Chatbot
# --------------------------------------------------
def initialize_chatbot():
    if st.session_state.qa_chain is None:
        with st.spinner("Loading chatbot..."):
            documents = [Document(page_content=HOSTEL_INFO)]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = splitter.split_documents(documents)

            embeddings = load_embeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)

            prompt = PromptTemplate(
                template="""
Answer the question using the context below.
If the answer is not in the context, say you don't have that information.

Context:
{context}

Question:
{question}

Answer:
""",
                input_variables=["context", "question"]
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
                chain_type_kwargs={"prompt": prompt}
            )

            st.session_state.qa_chain = qa_chain

# --------------------------------------------------
# Get Chatbot Response
# --------------------------------------------------
def get_response(question):
    return st.session_state.qa_chain.run(question)

# --------------------------------------------------
# Main App
# --------------------------------------------------
def main():
    st.title("🏠 Hostel Chatbot")
    st.markdown("Ask me anything about the hostel.")

    with st.sidebar:
        st.header("Sample Questions")
        st.markdown("""
- What room types are available?
- What are the meal timings?
- What facilities do you have?
- How do I book a room?
""")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    initialize_chatbot()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Type your question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_response(user_input)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
