from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma
from langchain_classic import hub

from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_ollama import OllamaLLM
from langchain_upstage import UpstageEmbeddings
from langchain_ollama import OllamaEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables.history import RunnableWithMessageHistory


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
  UPSTAGE = "embedding-query"
  QWEN3_8B = "qwen3-embedding:8b"
  QWEN3_4B = "qwen3-embedding:4b"

llm = None
embedding = None

def get_retriever():

  #print('get_retriever', embedding, embedding.model)
  
  if embedding == EmbeddingModel.UPSTAGE:
    collection_name = "gwp2025_pdf"
  else:
    collection_name = f"gwp2025_{embedding.model.replace(':', '_')}"

  #print('collection_name', collection_name) 

  database = Chroma(
    collection_name=collection_name,
    persist_directory="./chroma_pdf_db",
    embedding_function=embedding,
  )
  retriever = database.as_retriever(search_kwargs={"k": 5})
  #retriever = database.as_retriever()

  #print('retriever', retriever)

  return retriever

def get_dictionary_chain():
  dictionary = ["맞복 -> 맞춤형복지"]
  prompt = ChatPromptTemplate.from_template(f"""
    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
    만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
    그런 경우에는 질문만 리턴해 주세요
    사전:{dictionary}
    
    사용자의 질문:{{question}}
    """
  )

  dictionary_chain = prompt | llm | StrOutputParser()

  return dictionary_chain


def get_qa_chain():

  #llm = get_llm()
  retriever = get_retriever()
  prompt = hub.pull("rlm/rag-prompt")

  print('llm', llm)
  print('retriever', retriever)
  print('prompt', prompt)

  qa_chain = (
    RunnableParallel(
      context=retriever,
      question=RunnablePassthrough(),
    )
    | prompt
    | llm
    #| StrOutputParser()
  )

  return qa_chain

  def get_rag_chain():

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversation_rag_chain = RunnableWithMessageHistory(
      rag_chain,
      get_session_history,
      input_messages_key="question",
      history_messages_key="chat_history",
      output_messages_key="answer",
    )

    return conversation_rag_chain


def get_embedding_model(embedding_model:EmbeddingModel=EmbeddingModel.UPSTAGE):
  if embedding_model == EmbeddingModel.UPSTAGE:
    return UpstageEmbeddings(model="embedding-query", api_key=os.environ.get("UPSTAGE_API_KEY"))
  elif embedding_model == EmbeddingModel.QWEN3_8B:
    return OllamaEmbeddings(model="qwen3-embedding")
  elif embedding_model == EmbeddingModel.QWEN3_4B:
    return OllamaEmbeddings(model="qwen3-embedding:4b")



def get_llm(llm_model:LLMModel=LLMModel.GPT_OSS_20B):


  print('llm_model', llm_model, os.environ.get("GOOGLE_API_KEY"))
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
    # timeout 300
    gpt_mini_llm = ChatOpenAI(model="gpt-5-mini", timeout=100)

    return gpt_mini_llm
  elif llm_model == LLMModel.GPT_5_NANO:
    # timeout 300
    gpt_nano_llm = ChatOpenAI(model="gpt-5-nano", timeout=100)

    return gpt_nano_llm


def get_ai_message(user_message:str, llm_model:LLMModel, embedding_model:EmbeddingModel=EmbeddingModel.QWEN3_4B):


  global llm
  global embedding
  embedding = get_embedding_model(embedding_model)
  llm = get_llm(llm_model)
  dictionary_chain = get_dictionary_chain()
  qa_chain = get_qa_chain()
  rag_chain = dictionary_chain | qa_chain
  qa_message = rag_chain.stream(user_message)

  return qa_message
