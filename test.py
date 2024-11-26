# DEVELOPED BY: ANURAG, DHAWAL, ANIMESH
# TECHNOLOGY: RETRIEVAL-AUGMENTED GENERATION (RAG)
# APIs USED: GEMINI (Google Generative AI)
# PREREQUISITES: 
#  - streamlit
#  - PyPDF2
#  - langchain
#  - langchain-community
#  - faiss-cpu
#  - spacy
#  - python-dotenv

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set environment flag for PyPDF2 compatibility
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize embeddings with SpaCy
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

# Function to extract text from uploaded PDF files
def pdf_read(pdf_files):
    text = ""
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    st.warning(f"Unable to extract text from page {page.number + 1}.")
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

# Function to split extracted text into smaller chunks
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a FAISS vector store
def vector_store(text_chunks):
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_db")
        st.success("Vector store created successfully.")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

# Function to generate sample questions from the extracted content
def generate_sample_questions(pdf_content):
    if not pdf_content.strip():
        return ["Unable to generate questions as no content was extracted."]
    
    # Example static questions for testing:
    return [
        "What is the main topic discussed in the PDF?",
        "Can you summarize the key points from the text?",
        "What insights can be drawn from the data?",
        "Are there any case studies or examples mentioned?",
        "What are the conclusions presented in the document?"
    ]

# Function to handle RAG-based conversational chain
def get_conversational_chain(tools, user_question):
    try:
        llm_gemini = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            api_key=os.getenv("GEMINI_API_KEY"),
            verbose=True
        )

        # Define the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the provided context to frame responses."),
            ("human", "{input}"),  # The user input
            ("placeholder", "{agent_scratchpad}"),  # Required placeholder for agent scratchpad
        ])

        # Create the agent with tools and prompt
        agent = create_tool_calling_agent(llm_gemini, [tools], prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[tools], verbose=True)

        # Provide default value for 'agent_scratchpad'
        response = agent_executor.invoke({
            "input": user_question,
            "agent_scratchpad": ""  # Default value
        })
        return response['output']
    except Exception as e:
        st.error(f"Error in conversational chain: {e}")
        return None

# Function to process user queries
def user_input(user_question):
    try:
        new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
        retriever = new_db.as_retriever()
        retrieval_chain = create_retriever_tool(retriever, "pdf_extractor", "This tool answers queries from the PDF.")
        return get_conversational_chain(retrieval_chain, user_question)
    except Exception as e:
        st.error(f"Error processing user input: {e}")
        return None

# Streamlit main application
def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("RAG-based Chat with PDF")

    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        response = user_input(user_question)
        if response:
            st.write("Reply: ", response)

    with st.sidebar:
        st.title("Menu:")
        pdf_files = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process"):
            if pdf_files:
                with st.spinner("Processing..."):
                    raw_text = pdf_read(pdf_files)
                    if raw_text.strip():
                        st.text_area("Extracted Text", raw_text[:1000], height=300)
                        text_chunks = get_chunks(raw_text)
                        vector_store(text_chunks)
                        st.success("Processing complete. You can now ask questions.")
                        st.write("Sample Questions:")
                        questions = generate_sample_questions(raw_text)
                        for question in questions:
                            st.write(f"- {question}")
                    else:
                        st.error("No text could be extracted from the provided PDFs.")
            else:
                st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
