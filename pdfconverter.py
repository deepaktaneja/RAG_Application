import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCeMzr8JS3Z0StscPs9_CG_93FSAX3DQEE"  # Replace with your actual key


def load_pdf_to_chroma():
    """Loads a PDF into ChromaDB using specified embeddings."""

    try:
        loader = PyPDFLoader("C:\\Users\\anmol\\Downloads\\test.pdf")
        pages = loader.load_and_split()
       # print(pages[3])
    # documents = loader.load()
    except FileNotFoundError:
        print(f"Error: PDF file not found at pdf_path")
        return None
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
   # for p in pages:
     #   context = "\n\n".join(str(p) )
      #  texts = text_splitter.split_text(context)
     #   print(texts)


    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")  # Use the Gecko embedding model
    except Exception as e:
        print(f"Error initializing Gemini embeddings: {e}. Falling back to Sentence Transformers.")
        embeddings = SentenceTransformerEmbeddings(model_name="gemini-pro")

        #embeddings = SentenceTransformerEmbeddings(model_name="gemini-pro")

    try:
        db = Chroma.from_documents(pages, embeddings)
        return db
    except Exception as e:
        print(f"Error creating Chroma database: {e}")
        return None


if __name__ == "__main__":
    # pdf_file = input("Enter the path to the PDF file: ")
    query = "create a summary"
   # use_gemini = input("Use Gemini Embeddings? (yes/no): ").lower() == "yes"

    db = load_pdf_to_chroma()

    if db:
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002")  # Use ChatGemini for chat-based interaction
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
            result = qa.invoke(query)
            print("\nAnswer:\n", result)
        except Exception as e:
            print(f"Error during question answering: {e}")
    else:
        print("Failed to process the PDF.")
