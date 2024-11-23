import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import os

# Load environment variables
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\dhawa\\Downloads\\client_secret_658973278087-i5bn1sn9khbglho6of29l1jjriud3qdd.apps.googleusercontent.com.json"
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                yield page.extract_text()
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")

# Split text into manageable chunks
def get_text_chunks(text_generator, chunk_size=5000, chunk_overlap=500):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for text in text_generator:
        yield from text_splitter.split_text(text)

# Create or update a FAISS vector store
def get_vector_store(text_chunks, embeddings, persist_directory="faiss_index", batch_size=100):
    if os.path.exists(persist_directory):
        vector_store = FAISS.load_local(persist_directory, embeddings)
    else:
        vector_store = FAISS(embeddings)

    batch = []
    for chunk in text_chunks:
        batch.append(chunk)
        if len(batch) >= batch_size:
            vector_store.add_texts(batch)
            batch.clear()

    # Add any remaining chunks
    if batch:
        vector_store.add_texts(batch)

    vector_store.save_local(persist_directory)

# Load the conversational chain
def get_conversational_chain():
    prompt_template = """Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, say 'answer is not available in the context'. Do not guess.

    Context:\n{context}\n
    Question:\n{question}\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Handle user input and query the vector store
def user_input(user_question, persist_directory="faiss_index"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local(persist_directory, embeddings)
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("Reply: ", response["output_text"])

# Streamlit UI
def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("RAG-based Chat with PDF")

    # Sidebar for uploading files and configuration
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
        chunk_size = st.number_input("Chunk size", value=5000, step=100)
        chunk_overlap = st.number_input("Chunk overlap", value=500, step=100)
        batch_size = st.number_input("Batch size", value=100, step=10)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                        text_generator = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(text_generator, chunk_size, chunk_overlap)
                        get_vector_store(text_chunks, embeddings, batch_size=batch_size)
                        st.success("Processing complete!")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.error("Please upload at least one PDF file.")

    # User input for asking questions
    user_question = st.text_input("Ask a question from the PDF files")
    if user_question:
        with st.spinner("Generating response..."):
            try:
                user_input(user_question)
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
