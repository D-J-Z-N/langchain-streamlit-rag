from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
import openai

load_dotenv()

# For GitHub models (completions)
github_token = os.getenv("GITHUB_TOKEN")
github_endpoint = "https://models.github.ai/inference"
github_model = "openai/gpt-4.1-nano"

# For OpenAI models (embeddings)
openai_api_key = os.getenv("OPENAI_API_KEY")
embedding_model = "text-embedding-3-small"

# Load from web
web_loader = WebBaseLoader(
    web_paths=(
        "https://lt.wikipedia.org/wiki/Klaip%C4%97da",
        "https://klaipedatravel.lt/",
    ),
)
web_docs = web_loader.load()

# Load from the local text file
file_loader = PyPDFLoader("./Pesciomis-po-Klaipeda-LT.pdf")
file_docs = file_loader.load_and_split()

# Combine all documents
docs = web_docs + file_docs
print(f"Loaded {len(docs)} document(s) from all sources.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
splits = text_splitter.split_documents(docs)
print(f"Created {len(splits)} text splits for embedding.")

# Set up OpenAIEmbeddings using OpenAI public API
embedding_fn = OpenAIEmbeddings(
    model=embedding_model,
    api_key=openai_api_key,
)

try:
    print("Creating in-memory vector store...")
    if not splits:
        raise ValueError("No document splits were created. Check if the web page content was properly loaded and split.")
    vectorstore = InMemoryVectorStore.from_documents(
        documents=splits,
        embedding=embedding_fn,
    )
    print("✅ In-memory vector store created.")
except openai.AuthenticationError:
    print("🔴 AUTHENTICATION ERROR: Invalid OpenAI API key")
    st.stop()
except Exception as e:
    print(f"🔴 Error: {e}")
    import traceback
    print(traceback.format_exc())
    st.stop()

retriever = vectorstore.as_retriever()

template = """You are an assistant for question-answering tasks. Your task is to answer questions strictly based on the provided context about Klaipėda.
Use the following pieces of retrieved context to answer the question.
If the context does not contain the answer, you must state that you don't know.
Do not use any external knowledge or information you were trained on. Your entire answer must be derived from the text provided in the 'Context' section.

Question: {question} 

Context: {context} 

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    print(docs)
    return "\n\n".join(doc.page_content for doc in docs)

st.title("Streamlit LangChain Demo")
st.markdown("### Klaipėda RAG")

def generate_response(input_text):
    # GitHub models for chat completions
    llm = ChatOpenAI(base_url=github_endpoint, temperature=0.7, api_key=github_token, model=github_model)

    print("Searching for relevant documents...")
    fetched_docs = vectorstore.similarity_search(input_text, k=3)
    print(f"✅ Retrieved {len(fetched_docs)} documents.")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )
    
    st.info(rag_chain.invoke(input_text))

    st.subheader("📚 Sources")
    for i, doc in enumerate(fetched_docs, 1):
        with st.expander(f"Source {i}"):
            st.write(f"**Content:** {doc.page_content}")

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "Ask something about Klaipėdą",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)