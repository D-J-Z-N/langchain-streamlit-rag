from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os
from langchain import hub

from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
import bs4  # BeautifulSoup for parsing HTML

from chromadb.config import Settings


load_dotenv()  # take environment variables

# from .env file
# Load environment variables from .env file

token = os.getenv("GITHUB_API_KEY")  # Replace with your actual token
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

loader = WebBaseLoader(
    web_paths=("https://lt.wikipedia.org/wiki/Klaip%C4%97da",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("mw-page-title-main", "mw-content-ltr", "mw-heading")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
splits = text_splitter.split_documents(docs)

embedding_fn = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://models.inference.ai.azure.com",
    api_key=token,
)

# Check if you want to rebuild or load vectorstore
PERSIST_DIR = ".chroma"
REBUILD_VECTORSTORE = not os.path.exists(PERSIST_DIR)

if REBUILD_VECTORSTORE:
    if not splits:
        raise ValueError("No document splits were created. Check if the web page content was properly loaded and split.")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_fn,
        persist_directory=PERSIST_DIR,
    )
    vectorstore.persist()
else:
    vectorstore = Chroma(
        persist_directory=".chroma",
        embedding_function=embedding_fn,
    )


retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    print(docs)
    return "\n\n".join(doc.page_content for doc in docs)

st.title("Streamlit LangChain Demo")
st.markdown("### Klaipėda RAG")

def generate_response(input_text):
    llm = ChatOpenAI(base_url=endpoint, temperature=0.7, api_key=token, model=model)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )
    
    
    st.info(rag_chain.invoke(input_text))

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "Ask something about Klaipėdą",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
