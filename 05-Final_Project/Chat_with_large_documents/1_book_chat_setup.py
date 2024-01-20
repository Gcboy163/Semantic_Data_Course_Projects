import re
import pinecone
from decouple import config
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Pinecone
from typing import List

embeddings_api = OpenAIEmbeddings(openai_api_key=config("OPENAI_API_KEY"))

pinecone.init(api_key=config("PINECONE_API_KEY"), environment="gcp-starter")
pinecone_index = "langchain-vector-store"

loader = PyPDFLoader("data/After-the-Cure.pdf")
data: List[Document] = loader.load_and_split()

page_texts: List[str] = [page.page_content for page in data]
#print(page_texts[6][:1000])

page_texts_fixed: List[str] = [re.sub(r"\t|\n", " ", page) for page in page_texts]
#print(page_texts_fixed[6][:1000])

vector_database = Pinecone.from_texts(
    page_texts_fixed, embeddings_api, index_name=pinecone_index
)