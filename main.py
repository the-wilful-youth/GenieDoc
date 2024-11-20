import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent

import os
os.environment["KMP_DUPLICATE_LIB_OK"]="TRUE"

embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader(pdf):
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

def get_conventional_chain(tools, ques):
    os.environ["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"]=os.getenv["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"]
    llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0, api_key=os.getenv("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"), verbose=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key="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")
    prompt = ChatPromptTemplate.from_messages([(
        "system",
        """You are a helpful assistant. Answer the questions as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context, just say "answer is not availible in the context", don't provide the wrong answer""",
    ), 
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),])
    
    tool = [tools]
    agent = create_tool_calling_agent(llm, tool, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
    response = agent_executor.invoke({"input": ques})
    print(response)
    st.write("Reply: ", response['output'])

def user_input(user_question):

    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)

    retriever = new_db.as_retriever()
    retrieval_chain = create_retriever_tool(retriever, "pdf_extractor", "This tool is to give answer to queries from the pdf.")
    get_conventional_chain(retrieval_chain, user_question)

def main():
    st.set_page_config("Chat PDF")
    st.header("RAG based Chat with PDF")

    user_question = st.text_input("Ask a question from the PDF file")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF file and click on Submit & Process button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Done!")

if __name__ == "__main__":
    main()