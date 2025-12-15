import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader

load_dotenv()

if __name__ == "__main__":
    print("----Ingesting-----")
    loader = TextLoader("/Users/ravikanthkarra/Documents/Py/PromptEngineering/langchain-learnings/rag-gist/mediumblog1.txt")
    document = loader.load()

    print("splitting")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(document)
    print(f".....Document split into {len(chunks)} chunks.....")

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    print(".....ingesting chunks into vector store.....")
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=os.environ["PINECONE_INDEX_NAME"])
    print(".....chunks ingested into vector store.....")