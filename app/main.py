import os
import sys
import glob
from typing import List, Annotated, TypedDict, Literal
from contextlib import asynccontextmanager
import re

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
from langchain_core.tools import tool  # [NEW] Tool 데코레이터

# LangGraph
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition  # [NEW] 도구 노드

# 문서 처리 및 벡터 DB
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# [NEW] OPC 클라이언트 임포트
from opc_client import IgnitionOpcClient

# --- [0. 설정] ---
load_dotenv()

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
DB_PATH = "./faiss_index"
LLM_MODEL_NAME = "qwen2.5:7b"

# Ignition OPC UA 주소 (기본값)
OPC_ENDPOINT = os.getenv("OPC_ENDPOINT", "opc.tcp://localhost:62541")
OPC_USER = os.getenv("OPC_USER", "Admin")
OPC_PASSWORD = os.getenv("OPC_PASSWORD", "P@ssw0rd")

opc_client = IgnitionOpcClient(OPC_ENDPOINT)

# --- [1. 도구(Tools) 정의] ---


@tool
async def read_ignition_tag(tag_path: str):
    """
    Ignition SCADA의 태그 값을 읽습니다.
    '현재 온도 알려줘', '상태 확인해줘' 같은 질문에 사용하세요.

    Args:
        tag_path: 읽을 태그의 전체 경로 (예: '[default]Tank/Temperature')
    """
    print(f"🛠️ [Tool] 태그 읽기 시도: {tag_path}")
    return await opc_client.read_tag(tag_path)


@tool
async def write_ignition_tag(tag_path: str, value: str):
    """
    Ignition SCADA의 태그에 값을 씁니다(제어).
    '설정값을 50으로 바꿔', '모터 켜' 같은 명령에 사용하세요.

    Args:
        tag_path: 쓸 태그의 전체 경로 (예: '[default]Tank/Setpoint')
        value: 변경할 값 (숫자나 문자열)
    """
    print(f"🛠️ [Tool] 태그 쓰기 시도: {tag_path} -> {value}")
    return await opc_client.write_tag(tag_path, value)


# 사용할 도구 목록
tools = [read_ignition_tag, write_ignition_tag]


# --- [2. Lifespan & Setup] ---
def langsmith_setup(project_name="Ignition-Agent-RAG"):
    if os.environ.get("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_PROJECT"] = project_name
        print(f"[System] LangSmith 추적 활성화: {project_name}")


langsmith_setup()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_retriever
    print("\n[System] 서버 초기화 중...")

    print(f"[System] 임베딩 모델 로드 중... (CUDA)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if os.path.exists(DB_PATH):
        print("[System] 벡터 DB 로딩 중...")
        try:
            vectorstore = FAISS.load_local(
                DB_PATH, embeddings, allow_dangerous_deserialization=True
            )
            global_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            print("[System] DB 로딩 완료.")
        except Exception as e:
            print(f"[Error] DB 로딩 실패: {e}")
    else:
        print("[System] ⚠️ 저장된 DB가 없습니다. (문서 검색 기능 비활성화)")

    yield
    print("[System] 서버 종료")


app = FastAPI(title="Ignition RAG Agent", lifespan=lifespan)


# --- [3. LangGraph 로직] ---


class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' or 'no'")


class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    documents: List[Document]
    force_tool: bool
    forced_tag_path: str


# 1. 문서 검색
def retrieve(state: GraphState):
    print("\n[1] 문서 검색")
    question = state["messages"][-1].content
    if global_retriever is None:
        return {"documents": []}

    docs = global_retriever.invoke(question)
    print(f" -> {len(docs)}개 문서 검색됨")
    return {"documents": docs}


# 2. 문서 평가
def grade_documents(state: GraphState):
    print("\n[2] 문서 평가")
    question = state["messages"][-1].content
    documents = state["documents"]

    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0, num_gpu=-1)
    parser = JsonOutputParser(pydantic_object=GradeDocuments)

    # 평가 프롬프트
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grader. Return JSON {{'binary_score': 'yes'}} if the document is relevant to the question, otherwise {{'binary_score': 'no'}}.",
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

    print(f" -> {len(filtered_docs)}개 문서 유효함")
    return {"documents": filtered_docs}


CMD_WORDS = ["켜", "꺼", "멈", "정지", "시작", "가동", "on", "off", "설정", "set"]
DEVICE_HINT = re.compile(r"\bFAN\d+\b", re.IGNORECASE)  # FAN1, fan2 같은 패턴

TAG_PATTERN = re.compile(r"(\[[^\]]+\][A-Za-z0-9_\-\/]+)")


# 도구 사용 강제 여부 판단
def detect_realtime_intent(state: GraphState):
    text = state["messages"][-1].content
    lowered = text.lower()

    # 1) 사용자가 태그를 직접 쓴 경우
    m = TAG_PATTERN.search(text)
    tag = m.group(1) if m else ""

    # 2) 제어 명령인지 판단
    is_cmd = any(w in lowered for w in CMD_WORDS)

    # 3) 장비 힌트(예: FAN1)라도 있으면 제어로 취급
    has_device = bool(DEVICE_HINT.search(text))

    force = is_cmd and (bool(tag) or has_device)

    return {"force_tool": force, "forced_tag_path": tag}


# 도구 사용 강제 여부 판단
def detect_realtime_intent(state: GraphState):
    text = state["messages"][-1].content
    has_tag = "[default]" in text

    forced = ""
    if has_tag:
        m = re.search(r"(\[default\][\w\/\-]+)", text)
        forced = m.group(1) if m else ""

    return {"force_tool": has_tag, "forced_tag_path": forced}


async def force_control(state: GraphState):
    text = state["messages"][-1].content.lower()

    # 태그가 있으면 그대로 쓰고
    tag = state.get("forced_tag_path") or ""

    # 없으면 “규칙 기반 매핑”으로 결정 (FAN1 → [default]FAN1/Status)
    if not tag and DEVICE_HINT.search(state["messages"][-1].content):
        dev = DEVICE_HINT.search(state["messages"][-1].content).group(0).upper()
        tag = f"[default]{dev}/Status"

    # 값 결정
    value = "OFF" if ("멈" in text or "정지" in text or "off" in text) else "ON"

    result = await opc_client.write_tag(tag, value)
    return {"messages": [AIMessage(content=f"[제어 실행]\n{result}")]}


# 3. RAG 답변 (문서 기반)
def generate_rag(state: GraphState):
    print("\n[3-A] RAG 답변 생성")
    documents = state["documents"]
    question = state["messages"][-1].content

    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0, num_gpu=-1, num_ctx=4096)

    system_prompt = (
        "You are a Data Center Expert. "
        "Answer the question strictly in **Korean**, based **only** on the provided [Context]. "
        "Do not fabricate information."
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


# 4. 일반 대화 및 도구 사용 (문서 없음)
def generate_chat(state: GraphState):
    print("\n[3-B] 일반 대화/도구 모드")
    messages = state["messages"]

    # 도구 바인딩 (Bind Tools)
    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0.1, num_gpu=-1)
    llm_with_tools = llm.bind_tools(tools)

    system_msg = SystemMessage(
        content=(
            "You are an Ignition SCADA Operator. "
            "NEVER answer tag current values from memory. "
            "If the user asks for any current/now/real-time value or status, you MUST call read_ignition_tag again. "
            "If the user wants to change values, use write_ignition_tag. "
            "Answer naturally in Korean."
        )
    )

    # LLM 호출 (도구 호출 여부 결정 포함)
    response = llm_with_tools.invoke([system_msg] + messages)
    return {"messages": [response]}


# 5. 라우팅 결정
def route_after_detect(state: GraphState):
    if state["documents"]:
        return "generate_rag"
    if state.get("force_tool"):
        return "force_control"
    return "generate_chat"


# --- [4. 그래프 구축] ---#
def build_graph():
    memory = MemorySaver()
    workflow = StateGraph(GraphState)

    # 노드 추가
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate_rag", generate_rag)
    workflow.add_node("generate_chat", generate_chat)
    workflow.add_node("detect_realtime_intent", detect_realtime_intent)
    workflow.add_node("force_control", force_control)
    workflow.add_node("tools", ToolNode(tools))

    # 엣지 연결
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("grade_documents", "detect_realtime_intent")

    # detect 결과 라우팅
    workflow.add_conditional_edges(
        "detect_realtime_intent",
        route_after_detect,
        {
            "generate_rag": "generate_rag",
            "generate_chat": "generate_chat",
            "force_control": "force_control",
        },
    )

    # force_control는 종료
    workflow.add_edge("force_control", END)

    # generate_chat에서 tool_call이 있으면 tools로
    workflow.add_conditional_edges(
        "generate_chat",
        tools_condition,
        {"tools": "tools", END: END},
    )

    # tools 실행 결과를 다시 generate_chat로 보내 최종 문장 생성
    workflow.add_edge("tools", "generate_chat")

    # RAG는 종료
    workflow.add_edge("generate_rag", END)

    return workflow.compile(checkpointer=memory)


app_graph = build_graph()


# --- [5. API Endpoint] ---
class QueryRequest(BaseModel):
    question: str
    thread_id: str = "default_user"


@app.post("/ask")
async def ask_rag(request: QueryRequest):
    print(f"\n[Request] Thread: {request.thread_id} | Q: {request.question}")

    # Memory 설정을 위한 config
    config = RunnableConfig(configurable={"thread_id": request.thread_id})

    # 초기 입력 메시지
    inputs = {"messages": [HumanMessage(content=request.question)]}

    # [수정됨] 비동기 도구(OPC UA)를 사용하므로 await ainvoke()를 써야 합니다.
    result = await app_graph.ainvoke(inputs, config=config)

    # 최종 답변 추출 (마지막 메시지)
    final_answer = result["messages"][-1].content

    # 소스 정리 (RAG 모드일 때만 존재)
    sources = []
    if "documents" in result and result["documents"]:
        sources = list(
            set([doc.metadata.get("source", "Unknown") for doc in result["documents"]])
        )

    print(f"[Response] 완료 (Sources: {len(sources)})")

    return {
        "question": request.question,
        "answer": final_answer,
        "sources": sources,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
