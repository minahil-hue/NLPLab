import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="Hostel Chatbot", page_icon="🏠")

HF_TOKEN = "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"

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

Contact:
Phone: +91-9876543210
Email: info@hostelhome.com
Address: 123 University Road, Rawalpindi, Punjab
"""

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

@st.cache_resource
def load_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-large",
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

def initialize_chatbot():
    if st.session_state.vectorstore is None:
        with st.spinner("Loading chatbot..."):
            try:
                embeddings = load_embeddings()
                documents = [Document(page_content=HOSTEL_INFO)]
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                chunks = text_splitter.split_documents(documents)
                vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.vectorstore = vectorstore
                llm = load_llm()
                prompt_template = """Answer the question based on the context below. 
If you don't know, say you don't have that information.

Context: {context}

Question: {question}

Answer:"""
                PROMPT = PromptTemplate(
                    template=prompt_template, 
                    input_variables=["context", "question"]
                )
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
                    return_source_documents=False,
                    chain_type_kwargs={"prompt": PROMPT}
                )
                st.session_state.qa_chain = qa_chain
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return False
    return True

def get_response(question):
    try:
        response = st.session_state.qa_chain.run(question)
        return response.strip()
    except:
        return "Sorry, I encountered an error. Please try again."

def main():
    st.title("🏠 Hostel Chatbot")
    st.markdown("Ask me anything about the hostel!")
    
    with st.sidebar:
        st.header("About")
        st.info("AI-powered chatbot for hostel questions")
        st.markdown("### Sample Questions")
        st.markdown("""
        - What room types are available?
        - What are the meal timings?
        - What facilities do you have?
        - How do I book a room?
        """)
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    if not initialize_chatbot():
        st.error("Failed to load chatbot.")
        return
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Type your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_response(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
```

4. Click "Commit new file"

### Step 3: Update requirements.txt

1. Click on `requirements.txt` in your repo
2. Click the pencil icon (✏️) to edit
3. **Replace everything** with:
```
streamlit
langchain
langchain-community
faiss-cpu
sentence-transformers
huggingface-hub
transformers
