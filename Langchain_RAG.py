import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("langchain").setLevel(logging.ERROR)

# Install dependencies
!pip install --upgrade langchain faiss-cpu sentence-transformers google-colab
!pip install langchain-core
!pip install pypdf
!pip install langchain-text-splitters
!pip install langchain_huggingface
!pip install langchain_community
!pip install langchain_groq

GROQ_API_KEY="api_key"

from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader

# doc_text = """
# Elon Musk is a technology entrepreneur and engineer known for founding SpaceX and Tesla.
# He was born on June 28, 1971, in Pretoria, South Africa.
# His major achievements include advancing space exploration and electric vehicles.
# Musk is also involved with Neuralink and The Boring Company.
# This document provides a brief overview of Musk's background and accomplishments.
# """

urls = [
    "https://arxiv.org/abs/1706.03762",
    "https://arxiv.org/abs/2005.14165"
]
loader = WebBaseLoader(urls)

document = loader.load()

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("https://arxiv.org/pdf/1706.03762.pdf")
documents = loader.load()
print("Document Loading done")

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_split = splitter.split_documents(documents)

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    show_progress=False  # ✅ disables the widget
)

from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs_split, embeddings)

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    max_tokens=1024,
    api_key=GROQ_API_KEY,
)

from langchain_classic.chains import RetrievalQA

retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,  # to get source documents with answers
    chain_type="stuff",            # concatenates retrieved docs into prompt
)

query = "What is the attention mechanism in transformers?"

result = qa_chain.invoke({"query": query})

print("Answer:", result['result'])
# print("\nSource Documents:")
# for doc in result['source_documents']:
#     print(f"- {doc.page_content}")

!git remote add origin https://github.com/AyaanK123/RAG_Langchain.git
!git branch -M main
!git push -u origin main


