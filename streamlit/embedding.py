from enum import Enum
import time
import os

from langchain_text_splitters import MarkdownHeaderTextSplitter

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_upstage import UpstageEmbeddings

from llm import EmbeddingModel

from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()

def load_markdown_file(file_path:str="./data/2025공무원보수업무지침-맞품형복지.md"):
  # 문서 읽기
  loader = UnstructuredMarkdownLoader(file_path)

  # chunking
  headers_to_split_on = [
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
  ]


  text_splitter = MarkdownHeaderTextSplitter(
      headers_to_split_on=headers_to_split_on,
  )
  document_list = loader.load_and_split(text_splitter)

  return document_list

def get_embedding_model(embedding_model:EmbeddingModel=EmbeddingModel.UPSTAGE):
  if embedding_model == EmbeddingModel.UPSTAGE:
    return UpstageEmbeddings(model="embedding-query", api_key=os.environ.get("UPSTAGE_API_KEY"))
  elif embedding_model == EmbeddingModel.QWEN3_8B:
    return OllamaEmbeddings(model="qwen3-embedding")
  elif embedding_model == EmbeddingModel.QWEN3_4B:
    return OllamaEmbeddings(model="qwen3-embedding:4b")

def store_document_to_chroma(document_list:list, embedding_model:EmbeddingModel=EmbeddingModel.UPSTAGE):
  embedding_model = get_embedding_model(embedding_model)

  print(embedding_model)
  print(embedding_model.model)

  collection_name = f"gwp2025_{embedding_model.model.replace(':', '_')}"
  persist_directory = "./chroma_md_db"
  chroma_db = Chroma.from_documents(document_list, 
    embedding_model, 
    collection_name=collection_name, 
    persist_directory=persist_directory)

  return chroma_db

if __name__ == "__main__":
  document_list = load_markdown_file()
  print(document_list, '=>', len(document_list)) 

  #chroma_db = store_document_to_chroma(document_list, EmbeddingModel.UPSTAGE)
  # 시간측정 start
  start_time = time.time()
  chroma_db = store_document_to_chroma(document_list, EmbeddingModel.QWEN3_8B)
  print(chroma_db._collection_name, chroma_db)
  end_time = time.time()
  print(f"Time taken: {end_time - start_time} seconds")

  # 시간측정 end
  start_time = time.time()
  chroma_db = store_document_to_chroma(document_list, EmbeddingModel.QWEN3_4B)
  print(chroma_db._collection_name, chroma_db)
  end_time = time.time()
  print(f"Time taken: {end_time - start_time} seconds")

