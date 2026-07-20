import os
from dotenv import load_dotenv
 
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv()

DB = 'chroma_db'

embedddings =  NVIDIAEmbeddings(
    model = "nvidia/nv-embedqa-e5-v5",
    api_key=os.getenv("NVIDIA_API_KEY")
)

print("loading vector  database")
vectorstore = Chroma(
    persist_directory= "chroma_db",
    embedding_function=embedddings
)

retriver = vectorstore.as_retriever(search_kwargs={'k':4})

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature= 0.2,
    api_key=os.getenv("GROQ_API")
)

prompt = ChatPromptTemplate.from_template(
    """you are an helpfull assistant . only on the basis of the context given
     below answer the questions . if you could not find any context give a clear  answer as "i could not find anyy 
     answer in the document".
     
     context : {context}
     
     question :{input}
     
     answer :"""

)

document_chain = create_stuff_documents_chain(llm , prompt)
rag_chain  = create_retrieval_chain(retriver,document_chain)

print("\n RAG Chatbot is ready !")

while True:
    user_input = input("you :")

    if user_input.lower() in ['exit','quit']:
        print("bye")
        break

    response = rag_chain.invoke({"input": user_input})
    print("Bot :" , response["answer"],"\n")