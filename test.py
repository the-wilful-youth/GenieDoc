import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import os

load_dotenv()

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "TRUE"

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

#PDF READING
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template="""Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the provided context, say 'answer is not available in the context'. Don't provide wrong answers.
    Context:\n {context}?\n
    Question: \n{question}\n
    
    Answer:
    
    """
    model=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    promt=PromptTemplate(template=prompt_template, input_variables=["context","question"])
    chain=load_qa_chain(model, chain_type="stuff",promt=promt)
    return chain


def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embeddings-001")

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()


    response = chain(
        {"input_document":docs, "question":user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("reply: ", response["output_text"])


def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("RAG-based Chat with PDF")

    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and click 'Submit & Process'", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.error("Please upload at least one PDF file.")


if __name__ == "__main__":
    main()
