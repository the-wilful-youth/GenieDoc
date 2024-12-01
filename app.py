import streamlit as st

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    st.write("🤔 **Answer:** ", response["output_text"])


# Define the main application function
def main():
    # Configure the page
    st.set_page_config(
        page_title="GenieDoc",
        page_icon="📄",
        layout="wide",
    )

    # Apply custom CSS for improved aesthetics
    st.markdown(
        """
        <style>
        /* Style the sidebar */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        /* Header styles */
        .css-10trblm {
            color: #343a40;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 0.5em 1em;
            border-radius: 5px;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }

        /* Text input placeholder color */
        input::placeholder {
            color: #adb5bd;
            font-style: italic;
        }

        /* Spinner style */
        .stSpinner {
            color: #007bff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Page header with title and description
    st.title("📄 Chat with Your PDF Files Using GenieDoc")
    

    # Sidebar for file upload and processing
    with st.sidebar:
        st.header("📂 Upload & Process")
        st.markdown(
            """
            Upload your PDF files, and click **Submit & Process** to analyze the content.
            """
        )
        pdf_docs = st.file_uploader(
            "Upload PDF Files",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing your files..."):
                    raw_text = get_pdf_text(pdf_docs)  # Replace with your function
                    text_chunks = get_text_chunks(raw_text)  # Replace with your function
                    get_vector_store(text_chunks)  # Replace with your function
                    st.success("🎉 Files processed successfully!")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")

    # Main section for user interaction
    st.subheader("❓ Ask Your Question")
    user_question = st.text_input(
        "Type your question below:",
        placeholder="e.g., What is the main conclusion in the document?"
    )

    if user_question:
        # Handle user input
        st.write("🔎 **Your Question:**", user_question)
        user_input(user_question)  # Replace with your function

        # Display placeholder for answer (you can replace this with the actual function)
        

# Ensure the application runs when executed
if __name__ == "__main__":
    main()
