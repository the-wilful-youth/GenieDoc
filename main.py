#DEVELOPED BY: ANURAG, DHAWAL, ANIMESH
#TECHNOLOGY: RETRIEVAL-AUGMENTED GENERATION(RAG)
#APIs USED: OPENAI, CLAUDEAI, GEMENI
#PREREQUISITES: 
#  -   streamlit
#  -   PyPDF2
#  -   langchain
#  -   langchain-community
#  -   langchain-openai
#  -   langchain-anthropic
#  -   faiss-cpu
#  -   spacy
#  -   python-dotenv

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
# from langchain_anthropic import ChatAnthropic
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent

import os

# Load environment variables from .env
load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000000, chunk_overlap=100000)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

def get_conversational_chain(tools, ques):
    # Use proper function call for os.getenv
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    openai_api_key = os.getenv("OPEN_AI_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not anthropic_api_key or not openai_api_key or not gemini_api_key:
        raise ValueError("API keys for Anthropic and/or OpenAI are missing. Please set them in your environment variables.")
    
    llm_gemini = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        api_key=gemini_api_key,
        verbose=True
    )

    # llm_openai = ChatOpenAI(
    #     model_name="gpt-3.5-turbo", 
    #     temperature=0, 
    #     api_key=openai_api_key,
    #     verbose=True
    # )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say 'answer is not available in the context'."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    agent = create_tool_calling_agent(llm_gemini, [tools], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[tools], verbose=True)
    response = agent_executor.invoke({"input": ques})
    st.write("Reply: ", response['output'])

def user_input(user_question):
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    retrieval_chain = create_retriever_tool(retriever, "pdf_extractor", "This tool answers queries from the PDF.")
    get_conversational_chain(retrieval_chain, user_question)

def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("RAG-based Chat with PDF")

    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF Files and click 'Submit & Process'", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_doc:
                with st.spinner("Processing..."):
                    raw_text = pdf_read(pdf_doc)
                    text_chunks = get_chunks(raw_text)
                    vector_store(text_chunks)
                    st.success("Done")
            else:
                st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
