import os
from langchain_community.document_loaders import GithubFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

load_dotenv()

# Read GitHub token and Google API key from environment variables
github_token = os.getenv("GITHUB_TOKEN")
google_api_key = os.getenv("GOOGLE_API_KEY")


if not github_token or not google_api_key:
    raise ValueError("Environment variables GITHUB_TOKEN and GOOGLE_API_KEY must be set")


def load_github_documents(repo, branch, file_filter=lambda file_path: file_path.endswith(".md")):
    try:
        loader = GithubFileLoader(
            repo=repo,
            branch=branch,
            access_token=github_token,
            github_api_url="https://api.github.com",
            file_filter=file_filter
        )
        documents = loader.load()
        if not documents:
            print(f"No documents loaded from {repo}")
            return None
        return documents
    except Exception as e:
        print(f"Error loading documents from GitHub: {e}")
        return None

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def create_embeddings(docs):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    except Exception as e:
        print(f"Error initializing Google embeddings: {e}. Falling back to Sentence Transformers.")
        embeddings = SentenceTransformerEmbeddings(model_name="gemini-pro")
    return embeddings

def create_vector_store(docs, embeddings):
    try:
        db = Chroma.from_documents(docs, embeddings)
        return db
    except Exception as e:
        print(f"Error creating Chroma database: {e}")
        return None

def create_github_embeddings(repo, branch="master"):
    documents = load_github_documents(repo, branch)
    if not documents:
        return None

    docs = split_documents(documents)
    embeddings = create_embeddings(docs)
    db = create_vector_store(docs, embeddings)
    return db

def query_vector_store(db, query):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002")
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
        result = qa.invoke(query)
        print("\nAnswer:\n", result)
    except Exception as e:
        print(f"Error during question answering: {e}")

if __name__ == "__main__":
    repo = "deepaktaneja/RAG_Application"
    #repo="octocat/Hello-World"
    db = create_github_embeddings(repo)

    if db:
        query = "What is the name of miserly man and why he needs to change his ways"
        query_vector_store(db, query)
    else:
        print("Failed to process the GitHub repository.")