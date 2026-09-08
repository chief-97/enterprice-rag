import os
from dotenv import load_dotenv
 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PDF = "Artificial_Intelligence_in_India_Future_Perspective.pdf"


print("loading oyur pdf")

loader = PyPDFLoader(PDF)
documents = loader.load()
print(f"total pages loaded : {len(documents)}")


print("making chunks....")
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 150
)

chunks = splitter.split_documents(documents)
print(f"total chunks : {len(chunks)}")



embeddings = NVIDIAEmbeddings(
    model = "nvidia/nemotron-3-embed-1b",
    api_key= os.getenv("NVIDIA_API_KEY")
)

print("embeddings is getting generated and is stored in database" )

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory='chroma_db'
)

print("\nDone !  vector database is created and document ed chunks aare loaded")

