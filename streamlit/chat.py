import streamlit as st


from llm import get_ai_message
from llm import LLMModel
from llm import EmbeddingModel

llm_model: LLMModel = LLMModel.GPT_5_MINI
embedding_model: EmbeddingModel = EmbeddingModel.QWEN3_8B

st.set_page_config(page_title="맞춤형복지 챗봇", page_icon=":robot_face:")

st.title("🤖 맞춤형복지 챗봇")
st.caption(f"맞춤형 복지 챗봇입니다. 궁금하신 점이 있으시면 질문해주세요.:({llm_model.value}, {embedding_model.value})")

if 'messages' not in st.session_state:
  st.session_state.messages = []

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.write(message["content"])

if user_question := st.chat_input(placeholder="맞춤형복지 제도에 대해 궁금한 점을 입력해주세요."):
  with st.chat_message("user"):
    st.write(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

  with st.spinner("답변을 준비중입니다..."):
    answer = get_ai_message(user_question, llm_model, embedding_model)

    with st.chat_message("assistant"):  
      full_answer = st.write_stream(answer)  # 마지막 청크만 스트리밍

    st.session_state.messages.append({"role": "assistant", "content": full_answer})