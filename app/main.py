import os
import sys
import glob
from typing import List, Annotated, TypedDict, Literal
from contextlib import asynccontextmanager

# FastAPI
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

# LangChain & Models
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

# LangGraph
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver  # [NEW] 메모리 저장소

# 문서 처리 및 벡터 DB
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# --- [0. 설정] ---
load_dotenv()

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
DB_PATH = "./faiss_index"
LLM_MODEL_NAME = "qwen2.5:7b"

global_retriever = None


# --- [1. LangSmith 설정] ---
def langsmith_setup(project_name="Ignition-RAG-Chatbot"):
    if os.environ.get("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_PROJECT"] = project_name
        print(f"[System] LangSmith 추적 활성화: {project_name}")


langsmith_setup()


# --- [2. Lifespan: DB 로드] ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_retriever
    print("\n[System] 서버 초기화 중...")

    print(f"[System] 임베딩 모델({EMBEDDING_MODEL_NAME}) 로드 중... (CUDA)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if os.path.exists(DB_PATH):
        print("[System] 저장된 벡터 DB 발견. 로딩 중...")
        try:
            vectorstore = FAISS.load_local(
                DB_PATH, embeddings, allow_dangerous_deserialization=True
            )
            global_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            print("[System] DB 로딩 완료.")
        except Exception as e:
            print(f"[Error] DB 로딩 실패: {e}")
    else:
        # (편의상 생략: 기존 문서 로딩 로직은 동일하게 유지하거나, 필요시 복구 가능)
        print(
            "[System] ⚠️ 저장된 DB가 없습니다. 문서가 필요하면 이전 코드로 DB를 생성해주세요."
        )

    yield
    print("[System] 서버 종료")


app = FastAPI(title="Hybrid RAG Chatbot", lifespan=lifespan)

# --- [3. RAG & Chat 로직] ---


class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' or 'no'")


# [변경] State에 messages(대화기록) 추가
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    documents: List[Document]


# 1. 문서 검색 노드
def retrieve(state: GraphState):
    print("\n[1] 문서 검색")
    # 대화 기록 중 가장 마지막 메시지(질문) 추출
    question = state["messages"][-1].content

    if global_retriever is None:
        return {"documents": []}

    docs = global_retriever.invoke(question)
    print(f" -> {len(docs)}개 문서 검색됨")
    return {"documents": docs}


# 2. 문서 평가 노드
def grade_documents(state: GraphState):
    print("\n[2] 문서 평가")
    question = state["messages"][-1].content
    documents = state["documents"]

    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0, num_gpu=-1)
    parser = JsonOutputParser(pydantic_object=GradeDocuments)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grader. If the document is relevant to the question, return JSON {{'binary_score': 'yes'}}. Otherwise {{'binary_score': 'no'}}.",
            ),
            ("human", "Doc: {document}\nQuestion: {question}"),
        ]
    )

    chain = prompt | llm | parser

    filtered_docs = []
    for doc in documents:
        try:
            score = chain.invoke({"question": question, "document": doc.page_content})
            if score.get("binary_score") == "yes":
                filtered_docs.append(doc)
        except:
            continue

    print(f" -> {len(filtered_docs)}/{len(documents)}개 문서 유효함")
    return {"documents": filtered_docs}


# 3. RAG 답변 생성 노드 (문서 기반)
def generate_rag(state: GraphState):
    print("\n[3-A] RAG 답변 생성 (문서 기반)")
    documents = state["documents"]
    messages = state["messages"]
    question = messages[-1].content

    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0, num_gpu=-1, num_ctx=4096)

    # RAG 전용 프롬프트
    system_prompt = (
        "You are a Data Center Expert. "
        "Answer the question strictly in **Korean**, based **only** on the provided [Context]. "
        "Do not fabricate information not found in the context."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": documents, "question": question})

    return {"messages": [AIMessage(content=response)]}


# 4. [NEW] 일반 대화 노드 (문서 없음)
def generate_chat(state: GraphState):
    print("\n[3-B] 일반 대화 모드 (문서 없음)")
    messages = state["messages"]

    llm = ChatOllama(
        model=LLM_MODEL_NAME, temperature=0.7, num_gpu=-1
    )  # 창의성을 위해 온도 약간 높임

    # 일반 대화용 시스템 프롬프트 추가
    system_msg = SystemMessage(
        content="You are a Data Center Expert. Engage in professional and natural conversations using fluent Korean."
    )

    # 대화 기록 전체를 LLM에 전달 (Memory 효과)
    response = llm.invoke([system_msg] + messages)

    return {"messages": [response]}


# 5. [NEW] 조건부 라우팅 함수
def route_decision(state: GraphState) -> Literal["generate_rag", "generate_chat"]:
    if not state["documents"]:
        print(" -> 관련 문서 없음. 일반 대화로 전환합니다.")
        return "generate_chat"
    else:
        print(" -> 관련 문서 있음. RAG 답변을 생성합니다.")
        return "generate_rag"


# --- [4. 그래프 구축] ---


def build_graph():
    # MemorySaver 인스턴스 생성 (In-memory 저장소)
    memory = MemorySaver()

    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate_rag", generate_rag)
    workflow.add_node("generate_chat", generate_chat)  # 일반 대화 노드

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # 조건부 엣지: 문서 유무에 따라 RAG vs Chat 분기
    workflow.add_conditional_edges(
        "grade_documents",
        route_decision,
        {"generate_rag": "generate_rag", "generate_chat": "generate_chat"},
    )

    workflow.add_edge("generate_rag", END)
    workflow.add_edge("generate_chat", END)

    # [핵심] checkpointer(memory) 등록
    return workflow.compile(checkpointer=memory)


app_graph = build_graph()

# --- [5. API] ---


class QueryRequest(BaseModel):
    question: str
    thread_id: str = "default_user"  # 대화 맥락을 구분하는 ID


@app.post("/ask")
async def ask_rag(request: QueryRequest):
    print(f"\n[Request] Thread: {request.thread_id} | Q: {request.question}")

    # Memory 설정을 위한 config
    config = RunnableConfig(configurable={"thread_id": request.thread_id})

    # 초기 입력 메시지
    inputs = {"messages": [HumanMessage(content=request.question)]}

    # 그래프 실행 (config 포함)
    result = app_graph.invoke(inputs, config=config)

    # 최종 답변 추출 (마지막 메시지)
    final_answer = result["messages"][-1].content

    # 소스 정리 (RAG 모드일 때만 존재)
    sources = []
    if "documents" in result and result["documents"]:
        sources = list(
            set([doc.metadata.get("source", "Unknown") for doc in result["documents"]])
        )

    print(f"[Response] 답변 완료 (Sources: {len(sources)})")

    return {
        "question": request.question,
        "answer": final_answer,
        "sources": sources,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
