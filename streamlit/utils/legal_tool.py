
import os
import re
import json
import logging
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# 로깅 설정
logger = logging.getLogger(__name__)

def load_legal_references():
  """
  legal_references.json 파일을 로드합니다.
  """
  # 현재 파일(legal_tool.py)의 위치는 streamlit/utils/ 내부라고 가정
  current_dir = os.path.dirname(os.path.abspath(__file__))
  # data 경로는 ../../data/2026/legal_references.json
  legal_ref_path = os.path.join(current_dir, "..", "..", "data", "2026", "legal_references.json")
  
  # 경로가 존재하지 않는 경우 llm.py와 동일한 레벨(streamlit/)에서 실행되는 경우를 대비해 경로 조정 시도
  if not os.path.exists(legal_ref_path):
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
