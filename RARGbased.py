from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    temperature=0.5,
    max_new_tokens=1024
)

modelchat = ChatHuggingFace(llm=llm)

llmembeddings = OllamaEmbeddings(
    model="llama3.2"
)

docs = [
    Document(page_content="Artificial Intelligence is transforming industries."),
    Document(page_content="Machine learning is a subset of AI."),
    Document(page_content="FAISS is used for similarity search.")
]

vectorstore = FAISS.from_documents(docs, llmembeddings)

vectorstore.save_local("faiss_index")

results = vectorstore.similarity_search("What is FAISS?", k=2)

for doc in results:
    print(doc.page_content)