from enum import Enum
import os
import time
import uuid
import logging
import threading

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv가 없으면 환경 변수만 사용 (Streamlit Cloud Secrets 사용)
    pass

from langchain_core.chat_history import InMemoryChatMessageHistory

from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma
from langchain_classic import hub

from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_ollama import OllamaLLM
from langchain_openai import OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain_ollama import OllamaEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import json
import re
logger = logging.getLogger(__name__)

from langchain_core.runnables.history import RunnableWithMessageHistory

from supabase import create_client, Client

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


class LLMModel(Enum):
  GPT_OSS_20B = 'gpt-oss:20b'
  KANANA_1_5_8B = 'kanana-1.5-8b'
  CLOVAX = 'clovax'
  QWEN3_30B = 'qwen3:30b'
  QWEN3_14B = 'qwen3:14b'
  QWEN3_8B = 'qwen3:latest'
  EXAONE4_0_32B = 'exaone4.0:32b'
  GEMINI_2_5_FLASH = 'gemini-2.5-flash'
  GEMINI_3_PRO = 'gemini-3-pro-preview'
  GPT_5_MINI = 'gpt-5-mini'
  GPT_5_NANO = 'gpt-5-nano'


class EmbeddingModel(Enum):
  OPENAI = "text-embedding-3-smal"
  UPSTAGE = "embedding-query"
  QWEN3_8B = "qwen3-embedding:8b"
  QWEN3_4B = "qwen3-embedding:4b"

class VectorStore(Enum):
  PINECONE = "pinecone"
  CHROMA = "chroma"


def get_embedding_model(embedding_model:EmbeddingModel=EmbeddingModel.UPSTAGE):
  """임베딩 모델 인스턴스를 반환합니다."""

  if embedding_model == EmbeddingModel.OPENAI:
    return OpenAIEmbeddings(model="text-embedding-3-small")
  elif embedding_model == EmbeddingModel.UPSTAGE:
    return UpstageEmbeddings(model="embedding-query")
  elif embedding_model == EmbeddingModel.QWEN3_8B:
    return OllamaEmbeddings(model="qwen3-embedding")
  elif embedding_model == EmbeddingModel.QWEN3_4B:
    return OllamaEmbeddings(model="qwen3-embedding:4b")



def get_llm(llm_model:LLMModel=LLMModel.GPT_OSS_20B):
  """LLM 인스턴스를 반환합니다."""

  logger.info(f'llm_model: {llm_model}')
  if llm_model == LLMModel.EXAONE4_0_32B:
    return OllamaLLM(model="ingu627/exaone4.0:32b")
  elif llm_model == LLMModel.GPT_OSS_20B:
    return OllamaLLM(model="gpt-oss:20b")
  elif llm_model == LLMModel.KANANA_1_5_8B:
    return OllamaLLM(model="coolsoon/kanana-1.5-8b")
  elif llm_model == LLMModel.CLOVAX:
    return OllamaLLM(model="joonoh/HyperCLOVAX-SEED-Text-Instruct-1.5B")
  elif llm_model == LLMModel.QWEN3_30B:
    return OllamaLLM(model="qwen3:30b")
  elif llm_model == LLMModel.QWEN3_14B:
    return OllamaLLM(model="qwen3:14b")
  elif llm_model == LLMModel.QWEN3_8B:
    return OllamaLLM(model="qwen3:latest")
  elif llm_model == LLMModel.GEMINI_2_5_FLASH:
    # timeout 300
    gemini_llm = GoogleGenerativeAI(model="gemini-2.5-flash", timeout=100)

    return gemini_llm
  elif llm_model == LLMModel.GEMINI_3_PRO:
    # timeout 300
    gemini_llm = GoogleGenerativeAI(model="gemini-3-pro-preview", timeout=100)

    return gemini_llm
  elif llm_model == LLMModel.GPT_5_MINI:
    # gpt-5-mini는 reasoning 모델이므로 reasoning 토큰 + completion 토큰을 모두 고려해야 함
    # reasoning에 1500 토큰을 사용하면 실제 답변이 생성되지 않을 수 있으므로 충분히 크게 설정
    # timeout 300, max_tokens를 충분히 크게 설정하여 reasoning 후 실제 답변도 생성되도록 함
    gpt_mini_llm = ChatOpenAI(model="gpt-5-mini", timeout=100, max_tokens=4000)

    return gpt_mini_llm
  elif llm_model == LLMModel.GPT_5_NANO:
    # timeout 300
    gpt_nano_llm = ChatOpenAI(model="gpt-5-nano", timeout=100)

    return gpt_nano_llm


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
  """세션 ID에 따른 대화 이력 저장소를 반환합니다."""
  if session_id not in store:
    store[session_id] = InMemoryChatMessageHistory()
  return store[session_id]


def get_retriever_chroma():
  """Chroma DB 기반의 Retriver를 반환합니다."""
  logger.info(f'get_retriever with embedding model: {embedding.model}')

  collection_name = f"welfare_manual_{embedding.model.replace(':', '_')}"
  persist_directory = os.path.join(os.getcwd(), "chroma_db")

  #print('collection_name', collection_name)

  logger.info(f"{persist_directory}/{collection_name}")

  database = Chroma(
    collection_name=collection_name,
    persist_directory=persist_directory,
    embedding_function=embedding,
  )
  retriever = database.as_retriever(search_kwargs={"k": 3})
  #retriever = database.as_retriever()

  logger.info(f'Retriever initialized for collection: {collection_name}')

  return retriever


def get_retriever_pinecone():
  """Pinecone DB 기반의 Retriver를 반환합니다."""
  logger.info(f'get_retriever with embedding model: {embedding.model}')

  pinecone_db = PineconeVectorStore(index_name="gwp", embedding=embedding)
  retriever = pinecone_db.as_retriever(search_kwargs={"k": 3})
  return retriever


def save_log_to_supabase(session_id, question, answer, model, latency, tokens, source_documents):
  """
  로그를 Supabase에 저장합니다.
  """
  logger.info(f'save_log_to_supabase started')

  # LLMModel enum인 경우 .value로 문자열 변환
  model_name = model.value if hasattr(model, 'value') else str(model)

  # Document 객체를 JSON serializable한 딕셔너리로 변환 (id, score, section 포함)
  serialized_sources = []
  for doc in source_documents:
    if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
      # Document 객체에서 id 추출 (metadata의 id를 우선 사용)
      doc_id = None
      if isinstance(doc.metadata, dict):
        doc_id = doc.metadata.get("id", doc.metadata.get("_id", None))
      # metadata에 id가 없으면 Document.id 속성 사용
      if not doc_id and hasattr(doc, 'id'):
        doc_id = doc.id
      
      # score 추출 (metadata에서)
      doc_score = None
      if isinstance(doc.metadata, dict):
        doc_score = doc.metadata.get("score", None)
      
      # section 추출 (metadata에서 section_path 우선, 없으면 section)
      doc_section = None
      if isinstance(doc.metadata, dict):
        doc_section = doc.metadata.get("section_path") or doc.metadata.get("section")
        # section이 리스트인 경우 문자열로 변환
        if isinstance(doc_section, list):
          doc_section = " > ".join(str(s) for s in doc_section)
      
      # id, score, section 저장
      serialized_sources.append({
        "id": doc_id,
        "score": doc_score,
        "section": doc_section
      })
    elif isinstance(doc, dict):
      # dict인 경우 id, score, section 추출
      section = doc.get("section_path") or doc.get("section")
      # section이 리스트인 경우 문자열로 변환
      if isinstance(section, list):
        section = " > ".join(str(s) for s in section)
      
      serialized_sources.append({
        "id": doc.get("id", doc.get("_id", None)),
        "score": doc.get("score", None),
        "section": section
      })
    else:
      # 다른 형태인 경우 id, score, section만
      serialized_sources.append({"id": None, "score": None, "section": None})

  # 토큰 정보 추출
  prompt_tokens = tokens.get("prompt_tokens", 0) if tokens else 0
  completion_tokens = tokens.get("completion_tokens", 0) if tokens else 0
  total_tokens = tokens.get("total_tokens", 0) if tokens else 0

  data = {
      "session_id": session_id,
      "question": question,
      "answer": answer,
      "model_name": model_name,
      "latency_seconds": latency,
      "prompt_tokens": prompt_tokens,
      "completion_tokens": completion_tokens,
      "total_tokens": total_tokens,
      
      # Document 객체를 딕셔너리로 변환하여 전달
      "retrieved_sources": serialized_sources 
  }
  
  # 비동기로 로그 저장 (별도 스레드에서 실행)
  def save_log_async():
    try:
      supabase.table("chat_logs").insert(data).execute()
      logger.info(f"로그 저장 성공: session_id={session_id}")
    except Exception as e:
      logger.error(f"로그 저장 실패: {e}")
  
  # 별도 스레드에서 로그 저장 실행
  thread = threading.Thread(target=save_log_async, daemon=True)
  thread.start()  


def load_legal_references():
  """
  legal_references.json 파일을 로드합니다.
  """
  current_dir = os.path.dirname(os.path.abspath(__file__))
  legal_ref_path = os.path.join(current_dir, "..", "data", "2026", "legal_references.json")
  
  try:
    with open(legal_ref_path, 'r', encoding='utf-8') as f:
      data = json.load(f)
    return data.get("legal_texts", [])
  except Exception as e:
    logger.error(f"법령 참조 파일 로드 실패: {e}")
    return []


def search_legal_reference(law_name: str = None, article: str = None, keyword: str = None) -> str:
  """
  법령 참조 데이터에서 법령명, 조항, 또는 키워드로 검색합니다.
  
  Args:
    law_name: 법령명 (예: "공무원 후생복지에 관한 규정", "국가공무원법")
    article: 조항 (예: "제2조", "제11조")
    keyword: 검색 키워드
  
  Returns:
    검색된 법령 조항들의 텍스트
  """
  legal_texts = load_legal_references()
  results = []
  
  for law in legal_texts:
    law_name_match = False
    if law_name:
      # 법령명 부분 일치 검색
      if law_name.lower() in law["law_name"].lower() or law["law_name"].lower() in law_name.lower():
        law_name_match = True
    else:
      law_name_match = True  # 법령명이 지정되지 않으면 모든 법령 검색
    
    if not law_name_match:
      continue
    
    for provision in law["provisions"]:
      match = False
      
      # 조항 검색
      if article:
        if article in provision["article"] or provision["article"] in article:
          match = True
      
      # 키워드 검색
      if keyword:
        if (keyword.lower() in provision["content"].lower() or 
            keyword.lower() in provision["article"].lower()):
          match = True
      
      # 법령명만 지정된 경우 모든 조항 포함
      if law_name and not article and not keyword:
        match = True
      
      if match:
        results.append({
          "law_name": law["law_name"],
          "article": provision["article"],
          "content": provision["content"]
        })
  
  if not results:
    return "검색된 법령 조항이 없습니다."
  
  # 결과 포맷팅
  formatted_results = []
  for result in results:
    formatted_results.append(
      f"【{result['law_name']} {result['article']}】\n{result['content']}"
    )
  
  return "\n\n".join(formatted_results)


def get_legal_reference_tool():
  """
  법령 조회를 위한 LangChain Tool을 반환합니다.
  """
  class LegalReferenceInput(BaseModel):
    law_name: str = Field(None, description="법령명 (예: '공무원 후생복지에 관한 규정', '국가공무원법', '공무원보수 등의 업무지침')")
    article: str = Field(None, description="조항 번호 (예: '제2조', '제11조', '제10장')")
    keyword: str = Field(None, description="검색할 키워드 (법령명, 조항, 내용에서 검색)")
  
  tool = StructuredTool.from_function(
    func=search_legal_reference,
    name="search_legal_reference",
    description="""법령, 지침, 예규를 조회하는 도구입니다. 
답변 내용 중에 관련 법령, 지침, 예규가 언급되거나 참조가 필요한 경우 이 도구를 사용하세요.
법령명, 조항, 또는 키워드로 검색할 수 있습니다.""",
    args_schema=LegalReferenceInput,
    return_direct=False
  )
  
  return tool


def extract_legal_references_from_text(text: str) -> list[dict]:
  """
  텍스트에서 법령명과 조항을 추출합니다.
  
  Returns:
    [{"law_name": "...", "article": "..."}, ...] 형태의 리스트
  """
  # 법령명 패턴 (예: "공무원 후생복지에 관한 규정", "국가공무원법", "공무원보수 등의 업무지침")
  law_name_patterns = [
    r"공무원 후생복지에 관한 규정",
    r"국가공무원법",
    r"공무원임용령",
    r"공무원 수당 등에 관한 규정",
    r"공무원보수 등의 업무지침",
    r"상법",
    r"보험업법",
    r"국가유공자 등 예우 및 지원에 관한 법률",
  ]
  
  # 조항 패턴 (예: "제2조", "제11조", "제10장")
  article_pattern = r"제\d+[조장항호]"
  
  references = []
  
  # 법령명과 조항을 함께 찾기
  for law_pattern in law_name_patterns:
    # 법령명 다음에 조항이 오는 패턴 찾기
    pattern = rf"({law_pattern})\s*({article_pattern})"
    matches = re.finditer(pattern, text)
    for match in matches:
      law_name = match.group(1)
      article = match.group(2)
      references.append({"law_name": law_name, "article": article})
  
  # 중복 제거
  seen = set()
  unique_refs = []
  for ref in references:
    key = (ref["law_name"], ref["article"])
    if key not in seen:
      seen.add(key)
      unique_refs.append(ref)
  
  return unique_refs


def add_legal_references_to_answer(answer: str) -> str:
  """
  답변에 법령 참조를 추가합니다.
  """
  # 답변에서 법령 참조 추출
  legal_refs = extract_legal_references_from_text(answer)
  
  if not legal_refs:
    return answer
  
  # 각 법령 참조에 대해 tool로 조회
  legal_footnotes = []
  for ref in legal_refs:
    try:
      legal_text = search_legal_reference(
        law_name=ref.get("law_name"),
        article=ref.get("article")
      )
      if legal_text and "검색된 법령 조항이 없습니다" not in legal_text:
        # 각주 형식으로 추가
        legal_footnotes.append(f"\n\n[각주: {ref['law_name']} {ref['article']}]\n{legal_text}")
    except Exception as e:
      logger.warning(f"법령 조회 실패 ({ref}): {e}")
  
  # 법령 참조를 답변에 추가
  if legal_footnotes:
    answer += "\n\n---\n**관련 법령 조항:**" + "".join(legal_footnotes)
  
  return answer


def get_fewshot_examples_text():
  """
  Few-shot 예시를 텍스트 형태로 반환합니다.
  """
  examples = [
    {
      "question": "단체보험의 구성은?",
      "answer": """단체보험은 기본항목(필수)과 기본항목(선택) 그리고 자율항목으로 구성됩니다.

|**구분**|**내용**|**구성**|
|---|---|---|
|**기본항목 (필수)**|공무원조직의 안정성을 위하여 전체 공무원이 의무적으로 선택하여야 하는 항목|**생명·상해보험**|
|**기본항목 (선택)**|운영기관의 장이 정책적 필요에 따라 설정하고 구성원이 의무적으로 선택하여야 하는 항목|**본인 및 가족의료비 보장보험**, 건강검진 등|
|**자율항목**|운영기관의 장이 필요에 따라 설정하고 개별 구성원이 자유롭게 선택할 수 있는 항목|건강관리, 자기계발, 여가활용, 가정친화|

*출처: 2026년도 맞춤형 복지 기관담당자 업무 매뉴얼 > 제4장 맞춤형복지 단체보험 운영 > 2. 단체보험 업무 처리기준 113p*"""
    },
    {
      "question": "건강검진 복지점수 지원내용은?",
      "answer": "건강검진 복지점수는 격년으로 350점이 배정되며, 본인 및 가족의 건강검진 용도로 사용 가능합니다. 암검진을 포함한 개인이 희망하는 모든 종합건강검진 항목에 대해 모든 병원 및 기관에서 사용할 수 있습니다. 단, 단순 외래진료 및 치료는 지원 대상이 아니지만, 위·대장내시경 수면비나 용종 제거 등 건강검진과 동반된 비용은 청구 가능합니다."
    }
  ]
  
  # 예시를 텍스트로 포맷팅
  examples_text = "\n\n".join([
    f"질문: {ex['question']}\n답변: {ex['answer']}"
    for ex in examples
  ])
  
  return examples_text


def get_dictionary_chain():
  """사전 기반 질문 변경 체인 (맞복 -> 맞춤형복지). 단순 문자열 치환으로 최적화."""

  logger.info('get_dictionary_chain started')

  # LLM 호출 없이 단순 문자열 치환으로 변경하여 성능 향상
  def normalize_question(input_data) -> str:
    """질문 정규화: 맞복 -> 맞춤형복지"""
    # 딕셔너리인 경우 "question" 키에서 추출, 문자열인 경우 그대로 사용
    if isinstance(input_data, dict):
      question = input_data.get("question", "")
    else:
      question = str(input_data)
    
    normalized = question.replace("맞복", "맞춤형복지")
    return normalized

  dictionary_chain = RunnableLambda(normalize_question)

  logger.info('get_dictionary_chain ended')

  return dictionary_chain


def get_qa_chain():

  logger.info('get_qa_chain started')

  #llm = get_llm()
  retriever = get_retriever_pinecone()
  prompt = hub.pull("rlm/rag-prompt")

  #print('llm', llm)
  #print('retriever', retriever)
  #print('prompt', prompt)

  qa_chain = (
    RunnableParallel(
      context=retriever,
      question=RunnablePassthrough(),
    )
    | prompt
    | llm
    #| StrOutputParser()
  )

  logger.info('get_qa_chain ended')

  return qa_chain




def get_rag_chain():
  """
  대화 이력 기반 RAG 체인 (History-Aware RAG)을 구성하고 반환합니다.
  
  History-Aware Question -> Retrieval -> Context + History + Question -> Answer
  
  성능 최적화: 대화 이력이 없거나 짧을 때는 History-Aware Question Transformation을 스킵합니다.
  """

  logger.info('get_rag_chain started')

  # 기본 retriever (History-Aware 없이 사용)
  base_retriever = get_retriever_pinecone()
  
  # 1. History-Aware Question Transformation (성능 최적화)
  # 프롬프트 간소화 및 빠른 응답을 위한 최적화
  contextualize_q_system_prompt = (
    "대화 이력이 있으면 최신 질문을 독립적인 검색 쿼리로 변환하세요. "
    "대화 이력이 없으면 현재 질문을 그대로 반환하세요. "
    "검색 쿼리만 출력하세요."
  )
  contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", contextualize_q_system_prompt),
      ("placeholder", "{chat_history}"), # RunnableWithMessageHistory에 의해 채워짐
      ("user", "{question}"),
    ]
  )
  
  # 질문 변환용 LLM: 빠른 응답을 위해 별도 LLM 인스턴스 사용 (max_tokens 제한)
  # gpt-5-mini는 reasoning 모델이므로 질문 변환에는 제한된 토큰 사용
  # 전역 llm 변수를 확인하여 모델 타입에 따라 최적화
  try:
    # llm이 ChatOpenAI 인스턴스인지 확인
    if isinstance(llm, ChatOpenAI):
      # 모델 이름 확인 (model 속성 또는 model_name 속성)
      model_name = getattr(llm, 'model', None) or getattr(llm, 'model_name', None) or "gpt-5-mini"
      # 질문 변환은 간단하므로 작은 max_tokens로 빠르게 처리
      # timeout도 줄여서 빠른 응답
      question_transform_llm = ChatOpenAI(
        model=model_name, 
        timeout=50, 
        max_tokens=100,
        temperature=0  # 일관된 빠른 응답
      )
      logger.info(f"질문 변환용 LLM 생성: {model_name} (max_tokens=100, timeout=50)")
    else:
      # 다른 모델은 기존 llm 사용
      question_transform_llm = llm
      logger.info("질문 변환용으로 기존 LLM 사용")
  except Exception as e:
    logger.warning(f"질문 변환용 LLM 생성 실패, 기존 llm 사용: {e}")
    question_transform_llm = llm
  
  # 질문 변환 체인: chat_history와 question을 받아 검색 쿼리(새로운 질문)를 생성
  # 성능 최적화: 간소화된 프롬프트와 제한된 토큰으로 빠른 응답
  history_aware_retriever = (
    contextualize_q_prompt 
    | question_transform_llm 
    | StrOutputParser() 
    | base_retriever
  )

  # 2. Answer Generation Prompt (RAG Prompt with Few-shot examples)
  # Few-shot 예시를 포함한 커스텀 RAG 프롬프트 생성
  #fewshot_examples_text = get_fewshot_examples_text()
  
  # RAG 프롬프트에 few-shot 예시와 context를 결합
  rag_prompt_template = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 

욕을하거나 모욕하는 경우에 대답을 거부하고 정중히 질문해 달라고 요청하세요
이제 다음 컨텍스트를 사용하여 질문에 답변하세요. 위의 예시 형식을 참고하여 답변하되, 컨텍스트에 있는 정보를 정확히 사용하세요.
컨텍스트의 마크다운 구조를 최대한 활용하시고 문장은 줄바꿈을 해서 보기좋게 표현해주세요
컨텍스트에 표가 있는 경우 표를 보여주시고, 표 caption도 보여주세요
표 내에서 줄바꿈이 필요한 경우 반드시 <br> 태그를 사용해주세요. 실제 줄바꿈(\n)을 사용하면 테이블 셀이 깨지므로 <br> 태그만 사용해주세요. 
마크다운 리스트 형식(예: `- 항목`)은 절대 제거하지 말고 반드시 유지해주세요. 
컨텍스트에 세미콜론(`;`)이 있는 경우, 세미콜론을 줄바꿈으로 변환하되 각 줄 앞에 `- `를 추가하여 마크다운 리스트 형식으로 표현해주세요. 
예를 들어 "항목1; 항목2; 항목3"은 "- 항목1\n- 항목2\n- 항목3"으로 변환해주세요.
내용중에 관련법령이나 지침, 예규, 영등이 언급되는 경우 해당 조항을 검색해서 각주로 표현해주세요
출처를 보여주시고 출처 형식은 title > section_path (page_number p) 형식으로 표현해주세요
페이지 번호가 없으면 보여주지 마세요
출처는 줄바꿈을 해주시고 출처가 여러개일때는 markdown numbered 리스트 형식으로 표현해주세요
(예: \n\n**출처**:\n1. 2026년도_맞춤형_복지_배정_업무매뉴얼 > 01 맞춤형 복지점수 운영 > 1. 맞춤형 복지 업무 연간 일정 (8p)")
가능한 3문장 이내로 대답하시고 마지막에 추가적인 정보가 있다면 정보를 추가해주세요

Context: {{context}}

Question: {{question}}

Answer:"""
  
  rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)

  # 3. Final Answer Generation Chain
  rag_chain = (
    RunnableParallel(
      # 'context'는 History-Aware Retriever의 검색 결과를 받습니다.
      context=history_aware_retriever, 
      # 'question'은 원본 사용자 질문을 그대로 전달합니다.
      question=RunnablePassthrough(), 
    )
    | rag_prompt
    | llm
    #| StrOutputParser()
  )

  # 4. History Management Wrapper (for multi-turn)
  # RAG 체인 전체를 RunnableWithMessageHistory로 래핑하여 대화 이력을 관리합니다.
  final_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
  )

  logger.info(f'RAG chain (History-Aware) configuration complete.')

  return final_rag_chain


llm = None
embedding = None

store: dict[str, InMemoryChatMessageHistory] = {}


def get_ai_message(user_message:str, 
                   llm_model:LLMModel, 
                   embedding_model:EmbeddingModel=EmbeddingModel.QWEN3_4B, 
                   session_id:str=str(uuid.uuid4())):
  """
  사용자의 메시지를 받아 전체 RAG 체인을 실행하고 스트리밍 답변을 반환합니다.

  Args:
    user_message: 사용자의 질문.
    session_id: 대화 이력을 위한 고유 세션 ID (멀티턴 대화 시 필수).
    llm_model: 사용할 LLM 모델.
    embedding_model: 사용할 임베딩 모델.
  
  Returns:
    tuple: (stream_generator, metadata_dict)
      - stream_generator: 스트리밍 응답 generator
      - metadata_dict: {"context": [...], "full_answer": "...", "tokens": {...}} 형태의 메타데이터
  """
  
  global llm
  global embedding

  # 모델 초기화
  embedding = get_embedding_model(embedding_model)
  llm = get_llm(llm_model)
  
  # 체인 구성
  dictionary_chain = get_dictionary_chain()
  history_aware_rag_chain = get_rag_chain()

  # dictionary_chain의 문자열 출력을 {"question": output_string} 딕셔너리로 변환합니다.
  question_formatter = RunnableLambda(lambda x: {"question": x})

  # 최종 실행 체인: 질문 정규화 -> RAG (대화 이력 포함)
  rag_chain = dictionary_chain | question_formatter | history_aware_rag_chain

  # 세션 ID를 설정에 추가
  config = {"configurable": {"session_id": session_id}}
  
  # RAG 체인 실행 (invoke만 사용)
  logger.info(f"Invoking RAG chain with session_id: {session_id} and question: {user_message}")
  
  # invoke로 실행하여 응답과 메타데이터를 한 번에 받음
  qa_message = rag_chain.invoke({"question": user_message}, config=config)
  
  # 디버깅: qa_message 타입과 내용 확인
  logger.info(f"qa_message type: {type(qa_message)}")
  logger.info(f"qa_message has content attr: {hasattr(qa_message, 'content')}")
  if hasattr(qa_message, 'content'):
    logger.info(f"qa_message.content length: {len(qa_message.content) if qa_message.content else 0}")
  
  # LLM 응답에서 토큰 정보 추출
  tokens_info = {}
  if hasattr(qa_message, 'response_metadata'):
    response_metadata = qa_message.response_metadata
    # OpenAI 형식: response_metadata['token_usage']
    if 'token_usage' in response_metadata:
      token_usage = response_metadata['token_usage']
      tokens_info = {
        'prompt_tokens': token_usage.get('prompt_tokens', 0),
        'completion_tokens': token_usage.get('completion_tokens', 0),
        'total_tokens': token_usage.get('total_tokens', 0)
      }
  
  # Google Gemini 형식: usage_metadata (AIMessage 속성)
  if not tokens_info and hasattr(qa_message, 'usage_metadata'):
    usage_metadata = qa_message.usage_metadata
    tokens_info = {
      'prompt_tokens': getattr(usage_metadata, 'input_tokens', 0),
      'completion_tokens': getattr(usage_metadata, 'output_tokens', 0),
      'total_tokens': getattr(usage_metadata, 'total_tokens', 0)
    }
  
  # 답변 내용 추출 (AIMessage 객체에서 content 추출)
  try:
    if hasattr(qa_message, 'content'):
      full_answer = qa_message.content or ""
    elif hasattr(qa_message, 'text'):
      full_answer = qa_message.text or ""
    else:
      # 딕셔너리나 다른 형태일 수도 있음
      full_answer = str(qa_message) if qa_message else ""
    
    logger.info(f"full_answer extracted, length: {len(full_answer)}")
    if not full_answer:
      logger.warning(f"full_answer is empty! qa_message: {qa_message}, type: {type(qa_message)}")
  except Exception as e:
    logger.error(f"Error extracting full_answer: {e}")
    full_answer = str(qa_message) if qa_message else ""
  
  # context는 빈 리스트로 설정 (별도 추출 필요 시 추가 구현)
  context_docs = []
  
  # metadata 설정
  metadata = {
    "context": context_docs,
    "full_answer": full_answer,
    "tokens": tokens_info
  }

  return qa_message, metadata