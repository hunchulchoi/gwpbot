from enum import Enum
import time
import logging
import os
import re

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata

from llm import EmbeddingModel
from logger_config import setup_logging

from langchain_chroma import Chroma

from dotenv import load_dotenv

# 애플리케이션 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)
load_dotenv()

_BR_TAG_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_EXCESS_BLANKLINES_RE = re.compile(r"\n{3,}")
_REFNUM_AFTER_WORD_RE = re.compile(r"([가-힣A-Za-z])(\d{1,6})(?=([)\].,!?;:\s]|$))")


def _normalize_breaks(text: str) -> str:
  """
  임베딩/검색 품질을 위해 HTML <br>를 개행으로 정규화합니다.
  - 표 셀 내부의 줄바꿈 의도는 유지하면서, 토큰 노이즈를 줄입니다.
  """
  if not text:
    return text
  text = _BR_TAG_RE.sub("\n", text)
  # 과도한 빈 줄은 2줄로 제한
  text = _EXCESS_BLANKLINES_RE.sub("\n\n", text)
  # 단어 뒤에 붙은 참조번호(예: "가능합니다2.")는 공백만 삽입해 분리
  text = _REFNUM_AFTER_WORD_RE.sub(r"\1 \2", text)
  return text.strip()


def load_and_split_markdown_files(file_paths: list[str]):
  """
  Markdown 파일들을 읽어서 Document List로 분할하여 반환합니다.
  """
  all_splits = []
  for file_path in file_paths:
    # UnstructuredMarkdownLoader는 헤더 정보를 메타데이터에 포함시켜 로드합니다.
    # mode="elements"는 헤더, 테이블 등을 별도의 Document로 분리합니다.
    loader = UnstructuredMarkdownLoader(file_path, mode="elements")
    docs = loader.load()
    # 표/문단 안의 <br> 등을 정리 (렌더링 목적이 아니라 RAG 목적)
    for d in docs:
      d.page_content = _normalize_breaks(d.page_content)
    all_splits.extend(docs)

  # 3. 2차 분할: 글자 수/토큰 기준 (물리적 분할)
  # 이미 쪼개진 문서들을 다시 세부적으로 나눕니다.
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1500,    # 자를 크기
      chunk_overlap=200   # 겹치는 구간
  )

  # ★ 중요: 여기서는 split_text가 아니라 split_documents를 씁니다.
  # 이전 단계의 메타데이터를 그대로 유지하면서 내용을 쪼개줍니다.
  final_splits = text_splitter.split_documents(all_splits)
  
  # ChromaDB에 저장하기 전에, 지원하지 않는 복합 메타데이터(리스트 등)를 제거합니다.
  final_splits = filter_complex_metadata(final_splits)
  
  # [확인용 코드] 필터링 후 첫 번째 문서의 메타데이터를 출력해봅니다.
  print("--- 필터링 후 메타데이터 확인 ---")
  print(final_splits[0].metadata)
  
  return final_splits


def get_embedding_model(embedding_model:EmbeddingModel=EmbeddingModel.UPSTAGE):
  """
  Embedding Model을 반환합니다.
  """
  if embedding_model == EmbeddingModel.UPSTAGE:
    return UpstageEmbeddings(model="embedding-query")
  elif embedding_model == EmbeddingModel.QWEN3_8B:
    return OllamaEmbeddings(model="qwen3-embedding")
  elif embedding_model == EmbeddingModel.QWEN3_4B:
    return OllamaEmbeddings(model="qwen3-embedding:4b")


def store_documents_to_chroma(document_list:list, persist_directory: str, embedding_model_type:EmbeddingModel=EmbeddingModel.UPSTAGE):
  """
  Document List를 Chroma에 저장합니다.
  """
  embedding_model = get_embedding_model(embedding_model_type)

  logger.info(f'embedding_model: {embedding_model}')


  # 컬렉션 이름을 더 명확하게 지정
  collection_name = f"welfare_manual_{embedding_model.model.replace(':', '_')}"

  logger.info(f'{persist_directory}/{collection_name}')

  chroma_db = Chroma.from_documents(document_list,
    embedding_model, 
    collection_name=collection_name, 
    persist_directory=persist_directory)
  
  return chroma_db, collection_name

def main():
  logger.info("임베딩 스크립트를 시작합니다.")
  # 현재 스크립트 파일의 절대 경로를 기준으로 경로 설정
  # 이렇게 하면 어느 위치에서 실행하더라도 경로 문제가 발생하지 않습니다.
  current_dir = os.path.dirname(os.path.abspath(__file__))
  data_dir = os.path.join(current_dir, "..", "data")
  chroma_db_dir = os.path.join(current_dir, "..", "chroma_db")

  # 처리할 파일 목록
  file_paths = [
    os.path.join(data_dir, "2026배정업무매뉴얼.md"),
    os.path.join(data_dir, "2025공무원보수업무지침-맞춤형복지.md")
  ]
  document_list = load_and_split_markdown_files(file_paths)
  logger.info(f"총 {len(document_list)}개의 문서 조각 생성됨")
  logger.info(f"DB 저장 위치: {chroma_db_dir}")

  # 사용할 임베딩 모델 목록
  embedding_models_to_test = [
    EmbeddingModel.UPSTAGE,
    #EmbeddingModel.QWEN3_8B,
    #EmbeddingModel.QWEN3_4B
  ]

  for model_type in embedding_models_to_test:
    logger.info(f"--- {model_type.name} 임베딩 시작 ---")
    start_time = time.time()
    chroma_db, collection_name = store_documents_to_chroma(document_list, chroma_db_dir, model_type)
    end_time = time.time()
    logger.info(f"컬렉션 '{collection_name}' 생성 완료")
    logger.info(f"소요 시간: {end_time - start_time:.2f} 초")
  logger.info("임베딩 스크립트를 종료합니다.")

if __name__ == "__main__":
  main()
