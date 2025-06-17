from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import  GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import pickle

load_dotenv()

with open('scraped_data.pkl','rb') as f :
  scrapped_data = pickle.load(f)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=50)
chunks = splitter.create_documents([scrapped_data])
vector_store = FAISS.from_documents(chunks, embedding_model)
vector_store.save_local("tds_index")
