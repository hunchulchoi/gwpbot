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

os.environ["GRPC_DNS_RESOLVER"] = "native"

from utils.supabase_utils import save_log_to_supabase, save_report_to_supabase
from utils.legal_tool import add_legal_references_to_answer

from langchain_core.chat_history import InMemoryChatMessageHistory

from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma
from langchain_classic import hub

from langchain_openai import ChatOpenAI
from langchain_google_genai import (
    GoogleGenerativeAI,
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_ollama import OllamaLLM
from langchain_openai import OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain_ollama import OllamaEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)

from langchain_core.runnables.history import RunnableWithMessageHistory

from supabase import create_client, Client

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


index_name: str = "gwp-gemini3"


class LLMModel(Enum):
    GPT_OSS_20B = "gpt-oss:20b"
    KANANA_1_5_8B = "kanana-1.5-8b"
    CLOVAX = "clovax"
    QWEN3_30B = "qwen3:30b"
    QWEN3_14B = "qwen3:14b"
    QWEN3_8B = "qwen3:latest"
    EXAONE4_0_32B = "exaone4.0:32b"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_3_PRO = "gemini-3-pro-preview"
    GEMINI_3_FLASH = "gemini-3-flash-preview"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_4o_MINI = "gpt-4o-mini"


class EmbeddingModel(Enum):
    GEMINI001 = "gemini-embedding-001"
    OPENAI = "text-embedding-3-smal"
    UPSTAGE = "embedding-query"
    QWEN3_8B = "qwen3-embedding:8b"
    QWEN3_4B = "qwen3-embedding:4b"


class VectorStore(Enum):
    PINECONE = "pinecone"
    CHROMA = "chroma"


def get_embedding_model(embedding_model: EmbeddingModel = EmbeddingModel.UPSTAGE):
    """임베딩 모델 인스턴스를 반환합니다."""

    if embedding_model == EmbeddingModel.OPENAI:
        return OpenAIEmbeddings(model="text-embedding-3-small")
    elif embedding_model == EmbeddingModel.UPSTAGE:
        return UpstageEmbeddings(model="embedding-query")
    elif embedding_model == EmbeddingModel.QWEN3_8B:
        return OllamaEmbeddings(model="qwen3-embedding")
    elif embedding_model == EmbeddingModel.QWEN3_4B:
        return OllamaEmbeddings(model="qwen3-embedding:4b")
    elif embedding_model == EmbeddingModel.GEMINI001:
        return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


def get_llm(llm_model: LLMModel = LLMModel.GPT_OSS_20B):
    """LLM 인스턴스를 반환합니다."""

    logger.debug(f"llm_model: {llm_model}")
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
        gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", timeout=100)

        logger.debug(f"gemini_llm: {gemini_llm}")

        return gemini_llm
    elif llm_model == LLMModel.GEMINI_3_PRO:
        # timeout 300
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview", timeout=100, transport="rest"
        )

        return gemini_llm
    elif llm_model == LLMModel.GEMINI_3_FLASH:
        # timeout 300
        gemini_llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", timeout=100)

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
    elif llm_model == LLMModel.GPT_4o_MINI:
        # timeout 300
        gpt_4o_mini_llm = ChatOpenAI(model="gpt-4o-mini", timeout=100)

        return gpt_4o_mini_llm


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    세션 ID에 따른 대화 이력 저장소를 반환합니다.
    최근 3턴(6개 메시지)만 유지하거나, 오래된 대화를 요약하여 기억합니다.
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    history = store[session_id]
    messages = history.messages

    # 최근 3턴(6개 메시지)만 유지: user + assistant = 1턴
    MAX_TURNS = 3
    MAX_MESSAGES = MAX_TURNS * 2  # 6개 메시지 (질문 3개 + 답변 3개)

    if len(messages) > MAX_MESSAGES:
        # 방법 1: 최근 메시지만 유지 (간단하고 빠름)
        # messages_to_keep = messages[-MAX_MESSAGES:]
        # history.clear()
        # for msg in messages_to_keep:
        #   history.add_message(msg)
        # logger.debug(f"대화 이력 제한: {len(messages)}개 -> {len(history.messages)}개 메시지로 축소")

        # 방법 2: 오래된 대화를 요약하여 기억 (더 많은 컨텍스트 유지)
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        # 오래된 메시지와 최신 메시지 분리
        old_messages = messages[:-MAX_MESSAGES]
        recent_messages = messages[-MAX_MESSAGES:]

        # 오래된 대화를 요약
        if old_messages and llm:
            try:
                # 대화 내용 추출
                conversation_text = ""
                for msg in old_messages:
                    if hasattr(msg, "content"):
                        role = (
                            "사용자" if isinstance(msg, HumanMessage) else "어시스턴트"
                        )
                        conversation_text += f"{role}: {msg.content}\n"

                # 요약 프롬프트
                summary_prompt = f"""다음 대화 내용을 간단히 요약해주세요. 주요 질문과 답변의 핵심만 2-3문장으로 요약하세요.

대화 내용:
{conversation_text}

요약:"""

                # LLM으로 요약 생성
                summary_response = llm.invoke(summary_prompt)
                summary_text = (
                    summary_response.content
                    if hasattr(summary_response, "content")
                    else str(summary_response)
                )

                # 요약을 시스템 메시지로 추가
                summary_msg = SystemMessage(content=f"[이전 대화 요약] {summary_text}")

                # 이력 재구성: 요약 + 최신 메시지
                history.clear()
                history.add_message(summary_msg)
                for msg in recent_messages:
                    history.add_message(msg)

                logger.debug(
                    f"대화 이력 요약: {len(messages)}개 -> 요약 1개 + 최신 {len(recent_messages)}개 메시지"
                )
            except Exception as e:
                logger.warning(f"대화 이력 요약 실패, 최근 메시지만 유지: {e}")
                # 요약 실패 시 최근 메시지만 유지
                messages_to_keep = messages[-MAX_MESSAGES:]
                history.clear()
                for msg in messages_to_keep:
                    history.add_message(msg)
                logger.debug(
                    f"대화 이력 제한: {len(messages)}개 -> {len(history.messages)}개 메시지로 축소"
                )
        else:
            # LLM이 없거나 오래된 메시지가 없으면 최근 메시지만 유지
            messages_to_keep = messages[-MAX_MESSAGES:]
            history.clear()
            for msg in messages_to_keep:
                history.add_message(msg)
            logger.debug(
                f"대화 이력 제한: {len(messages)}개 -> {len(history.messages)}개 메시지로 축소"
            )

    return history


def get_retriever_chroma():
    """Chroma DB 기반의 Retriver를 반환합니다."""
    logger.debug(f"get_retriever with embedding model: {embedding.model}")

    collection_name = f"welfare_manual_{embedding.model.replace(':', '_')}"
    persist_directory = os.path.join(os.getcwd(), "chroma_db")

    # print('collection_name', collection_name)

    logger.debug(f"{persist_directory}/{collection_name}")

    database = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding,
    )
    retriever = database.as_retriever(search_kwargs={"k": 3})
    # retriever = database.as_retriever()

    logger.debug(f"Retriever initialized for collection: {collection_name}")

    return retriever


def get_retriever_pinecone():
    """Pinecone DB 기반의 Retriver를 반환합니다."""
    logger.debug(f"get_retriever with embedding model: {embedding.model}")

    global index_name

    pinecone_db = PineconeVectorStore(index_name=index_name, embedding=embedding)
    retriever = pinecone_db.as_retriever(search_kwargs={"k": 4})
    return retriever


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

*출처: 2026년도 맞춤형 복지 기관담당자 업무 매뉴얼 > 제4장 맞춤형복지 단체보험 운영 > 2. 단체보험 업무 처리기준 113p*""",
        },
        {
            "question": "건강검진 복지점수 지원내용은?",
            "answer": "건강검진 복지점수는 격년으로 350점이 배정되며, 본인 및 가족의 건강검진 용도로 사용 가능합니다. 암검진을 포함한 개인이 희망하는 모든 종합건강검진 항목에 대해 모든 병원 및 기관에서 사용할 수 있습니다. 단, 단순 외래진료 및 치료는 지원 대상이 아니지만, 위·대장내시경 수면비나 용종 제거 등 건강검진과 동반된 비용은 청구 가능합니다.",
        },
    ]

    # 예시를 텍스트로 포맷팅
    examples_text = "\n\n".join(
        [f"질문: {ex['question']}\n답변: {ex['answer']}" for ex in examples]
    )

    return examples_text


def get_dictionary_chain():
    """사전 기반 질문 변경 체인 (맞복 -> 맞춤형복지). 단순 문자열 치환으로 최적화."""

    logger.debug("get_dictionary_chain started")

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

    logger.debug("get_dictionary_chain ended")

    return dictionary_chain


def get_qa_chain():

    logger.debug("get_qa_chain started")

    # llm = get_llm()
    retriever = get_retriever_pinecone()
    prompt = hub.pull("rlm/rag-prompt")

    # print('llm', llm)
    # print('retriever', retriever)
    # print('prompt', prompt)

    qa_chain = (
        RunnableParallel(
            context=retriever,
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        # | StrOutputParser()
    )

    logger.debug("get_qa_chain ended")

    return qa_chain


def get_rag_chain():
    """
    대화 이력 기반 RAG 체인 (History-Aware RAG)을 구성하고 반환합니다.

    History-Aware Question -> Retrieval -> Context + History + Question -> Answer

    성능 최적화: 대화 이력이 없거나 짧을 때는 History-Aware Question Transformation을 스킵합니다.
    """

    logger.debug("get_rag_chain started")

    # 기본 retriever (History-Aware 없이 사용)
    base_retriever = get_retriever_pinecone()

    # 1. History-Aware Question Transformation (성능 최적화)
    # 프롬프트 간소화 및 빠른 응답을 위한 최적화
    contextualize_q_system_prompt = (
        "대화 이력이 있으면 최신 질문을 독립적인 검색 쿼리로 변환하세요. "
        "대화 이력이 없으면 현재 질문을 그대로 반환하세요. "
        "대화 이력과 질문이 연관성이 없으면 현재 질문을 그대로 반환하세요."
        "검색 쿼리만 출력하세요."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            (
                "placeholder",
                "{chat_history}",
            ),  # RunnableWithMessageHistory에 의해 채워짐
            ("user", "{question}"),
        ]
    )

    # 질문 변환 체인: chat_history와 question을 받아 검색 쿼리(새로운 질문)를 생성
    # 성능 최적화: 간소화된 프롬프트와 제한된 토큰으로 빠른 응답
    history_aware_retriever = (
        contextualize_q_prompt | llm | StrOutputParser() | base_retriever
    )

    # 2. Answer Generation Prompt (RAG Prompt with Few-shot examples)
    # Few-shot 예시를 포함한 커스텀 RAG 프롬프트 생성
    # fewshot_examples_text = get_fewshot_examples_text()

    # RAG 프롬프트에 few-shot 예시와 context를 결합
    rag_prompt_template = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 

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
        # | StrOutputParser()
    )

    # 4. History Management Wrapper (for multi-turn)
    # RAG 체인 전체를 RunnableWithMessageHistory로 래핑하여 대화 이력을 관리합니다.
    final_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    logger.debug(f"RAG chain (History-Aware) configuration complete.")

    return final_rag_chain


llm = None
embedding = None

store: dict[str, InMemoryChatMessageHistory] = {}


def get_ai_message(
    user_message: str,
    llm_model: LLMModel,
    embedding_model: EmbeddingModel = EmbeddingModel.QWEN3_4B,
    session_id: str = str(uuid.uuid4()),
    _index_name: str = "gwp-gemini3",
):
    """
    사용자의 메시지를 받아 전체 RAG 체인을 실행하고 스트리밍 답변을 반환합니다.

    Args:
      user_message: 사용자의 질문.
      session_id: 대화 이력을 위한 고유 세션 ID (멀티턴 대화 시 필수).
      llm_model: 사용할 LLM 모델.
      embedding_model: 사용할 임베딩 모델.
      index_name: 사용할 pinecone index 이름.

    Returns:
      tuple: (stream_generator, metadata_dict)
        - stream_generator: 스트리밍 응답 generator
        - metadata_dict: {"context": [...], "full_answer": "...", "tokens": {...}} 형태의 메타데이터
    """

    global llm
    global embedding
    global index_name

    # 모델 초기화
    embedding = get_embedding_model(embedding_model)
    llm = get_llm(llm_model)
    index_name = _index_name

    # 체인 구성
    dictionary_chain = get_dictionary_chain()
    history_aware_rag_chain = get_rag_chain()

    # dictionary_chain의 문자열 출력을 {"question": output_string} 딕셔너리로 변환합니다.
    question_formatter = RunnableLambda(lambda x: {"question": x})

    # 최종 실행 체인: 질문 정규화 -> RAG (대화 이력 포함)
    rag_chain = dictionary_chain | question_formatter | history_aware_rag_chain

    # 세션 ID를 설정에 추가
    config = {"configurable": {"session_id": session_id}}

    # CallbackHandler 정의: 검색된 문서 캡처
    from langchain_core.callbacks import BaseCallbackHandler

    class RetrievalCallbackHandler(BaseCallbackHandler):
        def __init__(self, metadata):
            self.metadata = metadata

        def on_retriever_end(self, documents, **kwargs):
            # 검색된 문서들을 metadata에 저장
            self.metadata["context"] = documents
            logger.debug(f"Retrieved {len(documents)} documents via callback")

    # 스트리밍 제너레이터와 빈 metadata 반환 (실제 metadata는 스트리밍 완료 후 수집)
    metadata = {"context": [], "full_answer": "", "tokens": {}}

    # CallbackHandler 인스턴스 생성
    callback_handler = RetrievalCallbackHandler(metadata)

    # config에 callback 추가
    config["callbacks"] = [callback_handler]

    # RAG 체인 실행 (stream 사용)
    logger.debug(
        f"Streaming RAG chain with session_id: {session_id} and question: {user_message}"
    )

    # stream으로 실행하여 스트리밍 응답 반환
    stream_generator = rag_chain.stream({"question": user_message}, config=config)

    return stream_generator, metadata
