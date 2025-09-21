import os
import faiss
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA

# Set up your local directory for PDFs
PDF_DIR = "pdfs/"
os.makedirs(PDF_DIR, exist_ok=True)

# Load Gemma model and tokenizer
model_id = "google/gemma-2b-it" # You can use another Gemma model if you prefer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# Create a pipeline for text generation
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
)

# Initialize LLM for LangChain
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Initialize embedding model (using a compatible open-source model)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_documents(uploaded_files):
    """Loads, splits, and embeds PDF documents."""
    # Save uploaded files to the 'pdfs' directory
    for uploaded_file in uploaded_files:
        with open(os.path.join(PDF_DIR, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Load documents from the directory
    loader = PyPDFLoader(os.path.join(PDF_DIR, uploaded_files[0].name)) # Just loads the first file for simplicity
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Create and save the vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local("faiss_index")

def get_response(user_prompt):
    """Retrieves relevant docs and generates a response."""
    # Load the vector store from the local directory
    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local("faiss_index", embeddings)
        retriever = vector_store.as_retriever()
        
        # Create a RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
        )

        # Get response
        response = qa_chain.run(user_prompt)
        return response
    
    return "Please upload and process documents first."