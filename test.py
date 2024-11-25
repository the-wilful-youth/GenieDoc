import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import os

# Load environment variables
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize embeddings
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

# Parallel PDF reader for efficient text extraction
def pdf_read_parallel(pdf_docs):
    def extract_text(pdf):
        pdf_reader = PdfReader(pdf)
        return "".join(page.extract_text() or "" for page in pdf_reader.pages)

    with ThreadPoolExecutor() as executor:
        texts = list(executor.map(extract_text, pdf_docs))
    return "".join(texts)

# Split text into smaller chunks for vector storage
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Create or load vector store
def vector_store(text_chunks, save_path="faiss_db"):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(save_path)
    return vector_store

# Load or initialize FAISS retriever
def get_retriever(db_path="faiss_db"):
    if os.path.exists(db_path):
        return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True).as_retriever()
    raise FileNotFoundError("Vector database not found. Please process PDF files first.")

# Generate responses using the retriever
def get_response(retriever, question):
    # Define prompt for the assistant
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question as detailed as possible using the provided context. If the answer is not in the context, say 'answer is not available in the context'."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # Create the tool with the retriever
    retriever_tool = create_retriever_tool(retriever, "pdf_extractor", "This tool answers queries from the PDF.")
    
    # Log the raw response for debugging
    response = retriever_tool.invoke({"query": question})
    #st.write("Debug: Raw Response from invoke()", response)
    
    # Handle response dynamically
    if isinstance(response, dict):
        if 'output' in response:
            return response['output']
        elif 'result' in response:
            return response['result']
        else:
            raise ValueError(f"Unexpected dictionary format: {response}")
    elif isinstance(response, str):
        return response  # Assume the response itself is the output
    else:
        raise ValueError(f"Unexpected response type: {type(response)} with content: {response}")

# Main application logic
def main():
    st.set_page_config(page_title="Enhanced Chat PDF", layout="wide")
    st.header("Efficient RAG-based Chat with PDF")

    # User input for querying
    user_question = st.text_input("Ask a question about the uploaded PDFs:")
    if user_question:
        try:
            retriever = get_retriever()
            response = get_response(retriever, user_question)
            st.write("Reply: ", response)
        except FileNotFoundError as e:
            st.error(str(e))
        except ValueError as e:
            st.error(f"Value Error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDFs and process", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = pdf_read_parallel(pdf_docs)  # Parallel PDF reading
                    text_chunks = get_chunks(raw_text)  # Split text into chunks
                    vector_store(text_chunks)  # Create vector store
                    st.success("Processing Complete!")
            else:
                st.error("Please upload at least one PDF file.")

# Run the app
if __name__ == "__main__":
    main()
