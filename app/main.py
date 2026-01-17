import os
import re
from typing import List, Annotated, TypedDict, Literal, Any, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
from opc_client import IgnitionOpcClient

# ----------------------------
# [0] 설정 및 초기화
# ----------------------------
load_dotenv()

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
DB_PATH = "./faiss_index"
LLM_MODEL_NAME = "llama3.1"

OPC_ENDPOINT = os.getenv("OPC_ENDPOINT", "opc.tcp://localhost:62541")
opc_client = IgnitionOpcClient(OPC_ENDPOINT)

SQL_HOST = os.getenv("SQL_HOST", "127.0.0.1")
SQL_PORT = int(os.getenv("SQL_PORT", "3306"))
SQL_USER = os.getenv("SQL_USER", "ignition")
SQL_PASSWORD = os.getenv("SQL_PASSWORD", "password")
SQL_DB = os.getenv("SQL_DB", "ignition")

global_retriever = None


def build_db_uri() -> str:
    return f"mysql+pymysql://{SQL_USER}:{SQL_PASSWORD}@{SQL_HOST}:{SQL_PORT}/{SQL_DB}"


sql_db = SQLDatabase.from_uri(build_db_uri())


# ----------------------------
# [1] 도구 정의 (Tools)
# ----------------------------
# --- 1. OPC UA용 도구 ---
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
chat_tools_list = [read_ignition_tag, write_ignition_tag]


# --- 2. SQL용 (이력/DB) - 커스텀 도구 ---
@tool
def db_list_tables():
    """DB의 모든 테이블 목록을 조회합니다."""
    try:
        return sql_db.get_table_names()
    except Exception as e:
        return f"Error: {e}"


@tool
def db_get_schema(table_names: str):
    """특정 테이블의 스키마(컬럼 정보)를 조회합니다. (입력: 'table1, table2')"""
    try:
        if isinstance(table_names, list):
            table_names = ", ".join(table_names)
        return sql_db.get_table_info(table_names.split(","))
    except Exception as e:
        return f"Error: {e}"


@tool
def db_query(query: str):
    """SQL SELECT 쿼리를 실행합니다. 반드시 LIMIT를 포함하세요."""
    try:
        if any(x in query.lower() for x in ["update", "delete", "drop", "insert"]):
            return "Error: Read-only allowed."
        return sql_db.run(query)
    except Exception as e:
        return f"SQL Error: {e}"


sql_tools_list = [db_list_tables, db_get_schema, db_query]


# ----------------------------
# [2] Lifespan
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_retriever
    print("\n[System] 서버 초기화 중...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        if os.path.exists(DB_PATH):
            vectorstore = FAISS.load_local(
                DB_PATH, embeddings, allow_dangerous_deserialization=True
            )
            global_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            print("[System] 벡터 DB 로드 완료.")
        else:
            print("[System] DB 없음. RAG 제한됨.")
    except Exception as e:
        print(f"[Warning] 벡터 DB 실패: {e}")
    yield
    print("[System] 서버 종료")


app = FastAPI(title="Ignition Agent", lifespan=lifespan)


# ----------------------------
# [3] Router (키워드 제거 -> LLM 판단)
# ----------------------------


# 라우팅 카테고리 정의
class RouteResponse(BaseModel):
    destination: Literal["sql_search", "rag_search", "chat"] = Field(
        description="The target agent to route the user request to."
    )


class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    intent_category: str  # intent_category만 남김 (type/payload 등 복잡한거 제거)
    payload: str
    documents: List[Document]


# ----------------------------
# [4] Node Functions
# ----------------------------


def intent_router(state: GraphState):
    """
    [핵심] LLM을 사용하여 사용자의 의도를 3가지 중 하나로 분류합니다.
    - sql_search: DB, 역사, 통계, 로그
    - rag_search: 매뉴얼, 지식, 정의, 방법
    - chat: 실시간 값 조회, 제어, 일반 대화
    """
    print("🚦 [Router] 의도 분류 중...")
    question = state["messages"][-1].content

    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0, format="json")

    # 분류 프롬프트
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a smart router. Classify the user question into one of three categories:
        
        1. 'sql_search': Questions about **historical data**, trends, logs, averages, past events, or database queries. (e.g., "What was the average RPM yesterday?", "Show error logs from last week")
        2. 'rag_search': Questions asking for **definitions, manuals, troubleshooting guides, specifications**, or general knowledge. (e.g., "What is a Chiller?", "How to fix Error 505?", "Explain the pump structure")
        3. 'chat': Requests for **real-time values**, **control commands**, greetings, or general chat. (e.g., "What is the current temperature?", "Turn on the motor", "Hi there")

        Return ONLY a JSON object: {{"destination": "sql_search" | "rag_search" | "chat"}}
        """,
            ),
            ("human", "{question}"),
        ]
    )

    chain = prompt | llm | JsonOutputParser()

    try:
        result = chain.invoke({"question": question})
        destination = result.get("destination", "chat")
    except:
        destination = "chat"  # 파싱 실패 시 기본값

    print(f"🚦 [Router] Decision: {destination}")

    return {
        "intent_category": destination,
        "payload": question,  # payload는 그대로 질문 내용
    }


def retrieve_rag(state: GraphState):
    if not global_retriever:
        return {"documents": []}
    return {"documents": global_retriever.invoke(state["payload"])}


def generate_rag(state: GraphState):
    # 실제 구현 시에는 검색된 문서를 바탕으로 LLM 답변 생성 필요
    context = "\n".join([d.page_content for d in state.get("documents", [])])
    return {
        "messages": [AIMessage(content=f"[RAG 결과]\n참고문서:\n{context[:200]}...")]
    }


def generate_chat(state: GraphState):
    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0.1)
    llm_with_tools = llm.bind_tools(chat_tools_list)
    system_msg = SystemMessage(
        content="You are an Ignition SCADA Operator. Answer in Korean."
    )
    response = llm_with_tools.invoke([system_msg] + state["messages"])
    return {"messages": [response]}


def sql_generate(state: GraphState):
    """
    SQL 실행 및 결과 요약 에이전트
    """
    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0)
    llm_with_tools = llm.bind_tools(sql_tools_list)

    # [핵심 수정] Ignition DB 구조를 '강제로' 주입하는 프롬프트
    system_msg = SystemMessage(
        content=(
            "You are an expert on **Ignition Historian Databases (MariaDB)**.\n"
            "This database uses a specific schema where Tag Names and Data are separated.\n"
            "You must follow the **Strict Execution Path** below. Do NOT guess table names.\n\n"
            "### 🗺️ Database Structure Map (READ CAREFULLY)\n"
            "1. **`sqlth_te` Table**: Contains Tag Definitions.\n"
            "   - Columns: `id` (Tag ID), `tagpath` (Tag Name)\n"
            "   - Usage: Query this table FIRST to convert a Tag Name (e.g., 'FAN1') into an `id`.\n"
            "2. **`sqlt_data_X_YYYY_MM` Tables**: Contains History Data (Partitioned by Month).\n"
            "   - Example: `sqlt_data_1_2026_01` (Data for Jan 2026)\n"
            "   - Columns: `tagid` (Foreign Key), `intvalue`, `floatvalue`, `t_stamp` (Unix Timestamp)\n"
            "   - Usage: Query this table SECOND using the `tagid` found in step 1.\n\n"
            "### 🛣️ Strict Execution Path\n"
            "When the user asks: 'Get average RPM of FAN1 on 2026-01-18':\n"
            "1. **Call `db_list_tables()`**: Find the partition table that matches the target date (look for `_2026_01`).\n"
            "2. **Call `db_query()` on `sqlth_te`**: Find the ID for the tag.\n"
            "   - Query: `SELECT id, tagpath FROM sqlth_te WHERE tagpath LIKE '%FAN1%'`\n"
            "3. **Call `db_query()` on partition table**: Use the `id` from Step 2 to get data.\n"
            "   - Query: `SELECT AVG(floatvalue) FROM sqlt_data_1_2026_01 WHERE tagid = [FOUND_ID] AND t_stamp BETWEEN ...`\n"
            "4. **Final Answer**: Summarize in Korean.\n\n"
            "**🚫 PROHIBITED ACTIONS:**\n"
            "- NEVER try `SELECT ... FROM FAN1`. 'FAN1' is a value in `tagpath`, NOT a table name.\n"
            "- NEVER skip `db_list_tables()`. You don't know which partition index (1, 5, etc.) exists for the date.\n"
        )
    )

    # 메시지 히스토리 포함
    messages = [system_msg] + state["messages"]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


async def exec_tag_read(state: GraphState):
    # Chat에서 Tool을 호출하면 이쪽으로 올 수도 있고, Chat Loop 내에서 처리될 수도 있음.
    # 여기서는 Chat Loop 사용하므로 이 노드는 사실상 안 쓰이거나 간단한 로그용
    pass


async def exec_tag_set(state: GraphState):
    pass


# ----------------------------
# [5] Graph Build
# ----------------------------


def route_decision(state: GraphState):
    return state["intent_category"]


def build_graph():
    memory = MemorySaver()
    wf = StateGraph(GraphState)

    # 노드 등록
    wf.add_node("intent_router", intent_router)  # [변경] ingest_intent 대신 Router 사용

    wf.add_node("retrieve_rag", retrieve_rag)
    wf.add_node("generate_rag", generate_rag)

    wf.add_node("generate_chat", generate_chat)
    wf.add_node("sql_generate", sql_generate)

    wf.add_node("chat_tools_node", ToolNode(chat_tools_list))
    wf.add_node("sql_tools_node", ToolNode(sql_tools_list))

    # 시작 -> 라우터
    wf.add_edge(START, "intent_router")

    # 라우터 -> 분기
    wf.add_conditional_edges(
        "intent_router",
        route_decision,
        {
            "sql_search": "sql_generate",
            "rag_search": "retrieve_rag",
            "chat": "generate_chat",
        },
    )

    # RAG 경로
    wf.add_edge("retrieve_rag", "generate_rag")
    wf.add_edge("generate_rag", END)

    # Chat 경로 (Loop)
    wf.add_conditional_edges(
        "generate_chat", tools_condition, {"tools": "chat_tools_node", END: END}
    )
    wf.add_edge("chat_tools_node", "generate_chat")

    # SQL 경로 (Loop)
    wf.add_conditional_edges(
        "sql_generate", tools_condition, {"tools": "sql_tools_node", END: END}
    )
    wf.add_edge("sql_tools_node", "sql_generate")

    return wf.compile(checkpointer=memory)


app_graph = build_graph()


# ----------------------------
# [6] API Endpoint
# ----------------------------
class QueryRequest(BaseModel):
    question: str
    thread_id: str = "default_user"


@app.post("/ask")
async def ask(request: QueryRequest):
    print(f"\nQ : {request.question}")

    inputs = {"messages": [HumanMessage(content=request.question)]}
    config = RunnableConfig(
        configurable={"thread_id": request.thread_id}, recursion_limit=30
    )

    result = await app_graph.ainvoke(inputs, config=config)

    last_message = result["messages"][-1]
    final_answer = (
        last_message.content
        if isinstance(last_message, AIMessage)
        else "답변을 생성하지 못했습니다."
    )

    return {
        "intent": result.get("intent_category"),
        "answer": final_answer,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
