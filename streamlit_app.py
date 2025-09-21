import streamlit as st
import rag_backend as rag # Assuming your RAG logic is in rag_backend.py

# --- Frontend UI ---
st.set_page_config(page_title="Gemma RAG Chatbot")
st.title("Local RAG Chatbot with Gemma")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle file uploads
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files for RAG",
        type=["pdf"],
        accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                rag.process_documents(uploaded_files)
            st.success("Documents processed!")

# Handle user prompt
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from the backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag.get_response(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})