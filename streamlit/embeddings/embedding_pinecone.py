from enum import Enum
import time
import logging
import os
os.environ["GRPC_DNS_RESOLVER"] = "native"
import re

from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata

from langchain_pinecone import PineconeVectorStore
from llm import EmbeddingModel, get_embedding_model
from logger_config import setup_logging

from langchain_chroma import Chroma

from dotenv import load_dotenv

# 애플리케이션 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)
load_dotenv()

# 임베딩 모델과 Pinecone 인덱스 이름
embedding_model_name:EmbeddingModel = EmbeddingModel.GEMINI001
index_name:str = "gwp-gemini3"

def load_and_split_json_files(file_paths: list[str], embedding_model:EmbeddingModel, index_name:str):
  """
  JSON 파일들을 읽어서 Pinecone에 저장합니다.
  """
  def metadata_func(record: dict, metadata: dict) -> dict:
    """
    JSONL 레코드에서 metadata 필드를 Document의 metadata로 변환합니다.
    """
    # record에서 metadata 필드 추출 (이미 dict 형태)
    if "metadata" in record and isinstance(record["metadata"], dict):
      # 기존 metadata와 병합 (record의 metadata가 우선)
      metadata.update(record["metadata"])
    # id도 metadata에 포함
    if "id" in record:
      metadata["id"] = record["id"]
    return metadata
  
  total_docs = 0
  for file_path in file_paths:
    logger.info(f"파일 로딩 중: {file_path}")
    # JSONL 파일 로드 (metadata_func를 사용하여 metadata 필드 포함)
    loader = JSONLoader(
      file_path, 
      json_lines=True, 
      jq_schema=".", 
      content_key="text",
      metadata_func=metadata_func
    )
    docs = loader.load()
    
    logger.info(f"로드된 문서 수: {len(docs)}개")
    store_documents_to_pinecone(docs, index_name, embedding_model)
    total_docs += len(docs)
    logger.info(f"Pinecone에 저장 완료: {len(docs)}개 문서")

  return total_docs


def store_documents_to_pinecone(document_list:list, index_name: str, embedding_model_name: EmbeddingModel):
  """
  Document List를 Pinecone에 저장합니다.
  """

  embedding_model = get_embedding_model(embedding_model_name)

  logger.info(f'embedding_model: {embedding_model}')

  pinecone_db = PineconeVectorStore(index_name=index_name, embedding=embedding_model)
  pinecone_db.add_documents(document_list)

  return pinecone_db


def main():
  global embedding_model_name
  global index_name

  logger.info("임베딩 스크립트를 시작합니다.")
  
  # 현재 스크립트 파일의 절대 경로를 기준으로 경로 설정
  # 이렇게 하면 어느 위치에서 실행하더라도 경로 문제가 발생하지 않습니다.
  current_dir = os.path.dirname(os.path.abspath(__file__))
  data_dir = os.path.join(current_dir, "..", "data", "2026")
  chroma_db_dir = os.path.join(current_dir, "..", "chroma_db")

  # 처리할 파일 목록
  file_paths = [
    os.path.join(data_dir, "2025년도공무원_보수등의_업무지침.jsonl"),
    os.path.join(data_dir, "2026년도_맞춤형_복지_기관담당자_업무매뉴얼.jsonl"),
    os.path.join(data_dir, "2026년도_맞춤형_복지_배정_업무매뉴얼.jsonl"),
  ]
  total_docs = load_and_split_json_files(file_paths, embedding_model_name, index_name)
  logger.info(f"총 {total_docs}개의 문서가 Pinecone에 저장되었습니다.")
  logger.info("임베딩 스크립트를 종료합니다.")

if __name__ == "__main__":
  main()
