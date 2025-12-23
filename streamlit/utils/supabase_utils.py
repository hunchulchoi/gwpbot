import os
import logging
import threading
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime
import json

# 로깅 설정 가져오기
logger = logging.getLogger(__name__)

# 환경 변수 로드
try:
    load_dotenv()
except ImportError:
    # dotenv가 없으면 환경 변수만 사용 (Streamlit Cloud Secrets 사용)
    pass

# Supabase 클라이언트 설정
# 전역 변수로 관리하여 재사용
_supabase_client = None


def get_supabase_client():
    """
    Supabase 클라이언트를 반환합니다. (싱글톤 패턴)
    """
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    try:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            logger.warning(
                "Supabase 환경변수(SUPABASE_URL, SUPABASE_KEY)가 설정되지 않았습니다."
            )
            return None

        _supabase_client = create_client(url, key)
        logger.info("Supabase 클라이언트 초기화 성공")
        return _supabase_client
    except Exception as e:
        logger.error(f"Supabase 클라이언트 초기화 실패: {e}")
        return None


def save_log_to_supabase(
    session_id, question, answer, model, latency, tokens, source_documents
):
    """
    대화 로그를 Supabase에 비동기적으로 저장합니다.
    """
    supabase = get_supabase_client()
    if not supabase:
        return

    try:
        # 소스 문서 정보 직렬화 (JSON 형태로 저장)
        serialized_docs = []
        if source_documents:
            for doc in source_documents:
                doc_info = {
                    "source": doc.metadata.get("source", "unknown"),
                    "page_content": (
                        doc.page_content[:200] + "..."
                        if len(doc.page_content) > 200
                        else doc.page_content
                    ),
                    "score": (
                        doc.metadata.get("score", 0) if hasattr(doc, "metadata") else 0
                    ),
                }
                serialized_docs.append(doc_info)

        # 모델명 문자열 변환
        model_name = str(model.value) if hasattr(model, "value") else str(model)

        data = {
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "model_name": model_name,
            "latency_seconds": float(latency),
            "prompt_tokens": tokens.get("prompt_tokens", 0) if tokens else 0,
            "completion_tokens": tokens.get("completion_tokens", 0) if tokens else 0,
            "total_tokens": tokens.get("total_tokens", 0) if tokens else 0,
            "retrieved_sources": serialized_docs,
        }

        # 동기로 로그 저장 및 ID 반환
        response = supabase.table("chat_logs").insert(data).execute()
        logger.debug(f"로그 저장 성공: session_id={session_id}")
        if response.data and len(response.data) > 0:
            return response.data[0]["id"]
        return None

    except Exception as e:
        logger.error(f"로그 저장 실패: {e}")
        return None


def save_report_to_supabase(chat_id, reason, details):
    """
    사용자 신고를 Supabase에 저장합니다.
    chat_id는 chat_logs 테이블의 id(integer)를 참조합니다.
    """
    supabase = get_supabase_client()
    if not supabase:
        logger.error("Supabase 클라이언트가 초기화되지 않아 신고를 저장할 수 없습니다.")
        return

    try:
        data = {
            "chat_id": chat_id,
            "reason": reason,
            "details": details,
        }

        logger.info(f"신고 저장 시도: {data}")
        response = supabase.table("feedback_reports").insert(data).execute()
        logger.info(f"신고 저장 성공: {response}")
    except Exception as e:
        logger.error(f"신고 저장 실패: {e}")
