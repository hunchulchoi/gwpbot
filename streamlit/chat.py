from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components

import uuid
import time
import logging
import re
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv가 없으면 환경 변수만 사용 (Streamlit Cloud Secrets 사용)
    pass

from llm import get_ai_message
from llm import LLMModel
from llm import EmbeddingModel
from llm import save_log_to_supabase
from llm import add_legal_references_to_answer
from logger_config import setup_logging

logger = logging.getLogger(__name__)


# 애플리케이션 로깅 설정
setup_logging()

llm_model: LLMModel = LLMModel.GPT_5_MINI
embedding_model: EmbeddingModel = EmbeddingModel.OPENAI




# session_id 초기화: 쿠키에서 device_id를 확인하거나 새로 생성
if 'session_id' not in st.session_state:
  # 쿼리 파라미터에서 device_id 확인 (JavaScript가 쿠키에서 읽어서 설정한 값)
  device_id = st.query_params.get('device_id')
  
  if device_id:
    # 쿼리 파라미터에 device_id가 있으면 사용 (쿠키에서 읽은 값)
    session_id = device_id
    st.session_state.session_id = session_id
  else:
    # 없으면 새 UUID 생성하고 쿠키에 저장하도록 JavaScript 실행
    new_uuid = str(uuid.uuid4())
    components.html(f"""
    <script>
      (function() {{
        // 쿠키에서 device_id 읽기
        function getCookie(name) {{
          const value = `; ${{document.cookie}}`;
          const parts = value.split(`; ${{name}}=`);
          if (parts.length === 2) return parts.pop().split(';').shift();
          return null;
        }}
        
        // 쿠키에 device_id 설정하기
        function setCookie(name, value, days = 365) {{
          const expires = new Date();
          expires.setTime(expires.getTime() + (days * 24 * 60 * 60 * 1000));
          // secure 플래그: HTTPS 연결에서만 쿠키 전송
          // httpOnly는 JavaScript에서 설정 불가 (서버에서만 설정 가능)
          const isSecure = window.location.protocol === 'https:';
          const secureFlag = isSecure ? ';secure' : '';
          document.cookie = `${{name}}=${{value}};expires=${{expires.toUTCString()}};path=/${{secureFlag}};SameSite=Lax`;
        }}
        
        let deviceId = getCookie('device_id');
        if (!deviceId) {{
          deviceId = '{new_uuid}';
          setCookie('device_id', deviceId, 365); // 1년간 유지
        }}
        
        // URL에 device_id를 쿼리 파라미터로 추가
        const url = new URL(window.location);
        if (!url.searchParams.has('device_id')) {{
          url.searchParams.set('device_id', deviceId);
          window.history.replaceState({{}}, '', url);
          // 페이지 리로드하여 쿼리 파라미터를 Streamlit에서 읽을 수 있게 함
          window.location.reload();
        }}
      }})();
    </script>
    """, height=0)
    # JavaScript가 리로드를 트리거하므로, 여기서는 임시로 새 UUID 사용
    session_id = new_uuid
    st.session_state.session_id = session_id
else:
  session_id = st.session_state.session_id

st.set_page_config(page_title="맞춤형복지 챗봇", page_icon=":robot_face:")

# 인증 상태 확인
if 'authenticated' not in st.session_state:
  st.session_state.authenticated = False
if 'failed_attempts' not in st.session_state:
  st.session_state.failed_attempts = 0
if 'blocked_until' not in st.session_state:
  st.session_state.blocked_until = None

# 비밀번호 확인 (처음 접속 시)
if not st.session_state.authenticated:
  current_time = time.time()
  
  # 접근 금지 시간 확인
  if st.session_state.blocked_until and current_time < st.session_state.blocked_until:
    remaining_time = int((st.session_state.blocked_until - current_time) / 60)  # 분 단위
    remaining_seconds = int((st.session_state.blocked_until - current_time) % 60)
    
    st.title("🔐 접근 금지")
    st.error(f"비밀번호를 여러 번 틀려서 접근이 금지되었습니다.")
    st.warning(f"접근 가능 시간까지 {remaining_time}분 {remaining_seconds}초 남았습니다.")
    st.stop()
  
  # 접근 금지 시간이 지났으면 초기화
  if st.session_state.blocked_until and current_time >= st.session_state.blocked_until:
    st.session_state.blocked_until = None
    st.session_state.failed_attempts = 0
  
  st.title("🔐 접근 권한 확인")
  st.info("이 챗봇을 사용하려면 비밀번호가 필요합니다.")
  
  if st.session_state.failed_attempts > 0:
    st.warning(f"비밀번호 입력 실패: {st.session_state.failed_attempts}회")
  
  password = st.text_input("비밀번호를 입력하세요", type="password", max_chars=7)
  
  # .env에서 비밀번호 가져오기
  correct_password = os.getenv("CHATBOT_PASSWORD", "8022912")  # 기본값은 8022912
  
  if st.button("확인"):
    if password == correct_password:
      # 비밀번호가 맞으면 인증 성공 및 실패 횟수 초기화
      st.session_state.authenticated = True
      st.session_state.failed_attempts = 0
      st.session_state.blocked_until = None
      st.rerun()
    else:
      # 비밀번호가 틀리면 실패 횟수 증가
      st.session_state.failed_attempts += 1
      
      # 3번 이상 틀리면 접근 금지
      if st.session_state.failed_attempts >= 3:
        # 접근 금지 시간 계산: 10분, 20분, 40분 (2배씩 증가)
        block_minutes = 10 * (2 ** (st.session_state.failed_attempts - 3))
        st.session_state.blocked_until = current_time + (block_minutes * 60)
        
        st.error(f"비밀번호를 {st.session_state.failed_attempts}회 틀려서 {block_minutes}분간 접근이 금지되었습니다.")
        st.rerun()
      else:
        st.error(f"비밀번호가 올바르지 않습니다. ({st.session_state.failed_attempts}/3회 실패)")
  
  st.stop()  # 인증 전에는 나머지 코드 실행 중단

st.title("🤖 맞춤형복지 챗봇")
st.caption(f"맞춤형 복지 챗봇입니다. 궁금하신 점이 있으시면 질문해주세요.:(powered by {llm_model.value})")

if 'messages' not in st.session_state:
  st.session_state.messages = []

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.write(message["content"])

if user_question := st.chat_input(placeholder="맞춤형복지 제도에 대해 궁금한 점을 입력해주세요."):
  # 답변 생성 시간 측정 시작
  start_time = time.time()
  
  with st.chat_message("user"):
    st.write(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

  with st.spinner("답변을 준비중입니다..."):
    qa_message, metadata = get_ai_message(user_question, llm_model, embedding_model, session_id)

    # 답변 생성 시간 계산
    end_time = time.time()
    latency = end_time - start_time
    
    # 답변 내용 추출
    full_answer = metadata.get("full_answer", "")
    if hasattr(qa_message, 'content'):
      full_answer = qa_message.content
    elif not full_answer:
      full_answer = str(qa_message)
    
    # <br> 태그 처리: 테이블 내에서는 HTML <br>로 유지, 테이블 외부에서는 줄바꿈으로 변환
    def process_br_tags(text):
      # 테이블 패턴 찾기 (|로 시작하거나 끝나는 줄)
      lines = text.split('\n')
      result_lines = []
      in_table = False
      
      for line in lines:
        stripped = line.strip()
        # 테이블 시작/종료 감지
        if '|' in line and (stripped.startswith('|') or stripped.endswith('|')):
          in_table = True
          # 테이블 내에서는 <br>을 HTML로 유지 (unsafe_allow_html로 렌더링)
          # 이미 <br> 태그가 있으면 그대로 유지
          line = re.sub(r'<br\s*/?>', '<br>', line, flags=re.IGNORECASE)
        elif in_table and stripped and '|' not in line:
          # 테이블 종료 (빈 줄이 아니고 |가 없는 줄)
          in_table = False
        elif not in_table:
          # 테이블 외부에서는 <br>을 줄바꿈으로 변환
          line = re.sub(r'<br\s*/?>', '\n', line, flags=re.IGNORECASE)
        
        result_lines.append(line)
      
      # 테이블 내부의 줄바꿈을 <br>로 변환 (테이블 셀 내부 줄바꿈 처리)
      # 전체 텍스트를 다시 분석하여 테이블 셀 내부의 줄바꿈을 <br>로 변환
      result_text = '\n'.join(result_lines)
      
      # 테이블 행 패턴: |로 시작하고 끝나는 줄
      table_row_pattern = re.compile(r'^(\s*\|[^|\n]*\|[^|\n]*\|\s*)$', re.MULTILINE)
      
      def replace_newlines_in_table_cells(match):
        row = match.group(1)
        # 셀 구분자 | 사이의 내용에서 줄바꿈을 <br>로 변환
        # 단, 이미 <br> 태그가 있으면 그대로 유지
        cells = row.split('|')
        processed_cells = []
        for i, cell in enumerate(cells):
          if i == 0 or i == len(cells) - 1:
            # 첫 번째와 마지막은 빈 문자열이거나 공백만 있음
            processed_cells.append(cell)
          else:
            # 셀 내부의 줄바꿈을 <br>로 변환 (단, 이미 <br>이 있으면 유지)
            if '\n' in cell and '<br' not in cell.lower():
              cell = cell.replace('\n', '<br>')
            processed_cells.append(cell)
        return '|'.join(processed_cells)
      
      # 테이블 행 내부의 줄바꿈을 <br>로 변환
      result_text = table_row_pattern.sub(replace_newlines_in_table_cells, result_text)
      
      return result_text
    
    # <br> 태그 처리
    processed_answer = process_br_tags(full_answer)
    full_answer = processed_answer
    
    with st.chat_message("assistant"):  
      # 답변 표시
      st.markdown(full_answer, unsafe_allow_html=True)
    
    # 법령 참조 추가 (답변에 법령명이나 조항이 있는 경우)
    full_answer_with_legal_refs = add_legal_references_to_answer(full_answer)
    
    # 법령 참조가 추가된 경우 답변 업데이트
    if full_answer_with_legal_refs != full_answer:
      st.markdown("---")
      st.markdown("**관련 법령 조항이 추가되었습니다.**")
      # 법령 참조 부분만 표시
      legal_refs_section = full_answer_with_legal_refs[len(full_answer):]
      st.markdown(legal_refs_section)
      full_answer = full_answer_with_legal_refs
    
    st.session_state.messages.append({"role": "assistant", "content": full_answer})

    st.caption(f"답변 생성 시간: {latency:.2f} 초 ({len(full_answer) / latency:.2f} 자/초) @{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 로그 저장 (스트리밍 완료 후)
    try:
      save_log_to_supabase(
        session_id=session_id,
        question=user_question,
        answer=full_answer,
        model=llm_model,
        latency=latency,
        tokens=metadata.get("tokens", {}),
        source_documents=metadata.get("context", [])
      )
    except Exception as e:
      logger.error(f"로그 저장 실패: {e}")