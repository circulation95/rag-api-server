from typing import List, Union
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonAstREPLTool
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks
from dotenv import load_dotenv
from rag.pdf import PDFRetrievalChain
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_core.prompts import ChatPromptTemplate

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Literal
from langchain import hub
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langchain_teddynote.models import get_model_name, LLMs
from langchain_core.tools.retriever import create_retriever_tool

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_teddynote.graphs import visualize_graph

from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import stream_graph, invoke_graph, random_uuid
from langchain_core.documents import Document
from langchain_teddynote.tools.tavily import TavilySearch

load_dotenv()
logging.langsmith("Adaptive RAG")

st.title("Adaptive RAG")

# 상수 정의
class MessageRole:
    """
    메시지 역할을 정의하는 클래스입니다.
    """

    USER = "user"  # 사용자 메시지 역할
    TOOL = "tool"  # 도구 메시지 역할
    ASSISTANT = "assistant"  # 어시스턴트 메시지 역할


class MessageType:
    """
    메시지 유형을 정의하는 클래스입니다.
    """

    TEXT = "text"  # 텍스트 메시지
    FIGURE = "figure"  # 그림 메시지
    CODE = "code"  # 코드 메시지
    DATAFRAME = "dataframe"  # 데이터프레임 메시지

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # 대화 내용을 저장할 리스트 초기화

if "graph" not in st.session_state:
    st.session_state["graph"] = None


# 사용자 쿼리를 가장 관련성 높은 데이터 소스로 라우팅하는 데이터 모델
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    # 데이터 소스 선택을 위한 리터럴 타입 필드
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


# 그래프의 상태 정의
class GraphState(TypedDict):
    """
    그래프의 상태를 나타내는 데이터 모델

    Attributes:
        question: 질문
        generation: LLM 생성된 답변
        documents: 도큐먼트 리스트
    """

    question: Annotated[str, "User question"]
    generation: Annotated[str, "LLM generated answer"]
    documents: Annotated[List[str], "List of documents"]

# 문서 평가를 위한 데이터 모델 정의
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# 할루시네이션 체크를 위한 데이터 모델 정의
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary scoring to evaluate the appropriateness of answers to questions"""

    binary_score: str = Field(
        description="Indicate 'yes' or 'no' whether the answer solves the question"
    )

def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f".cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    pdf = PDFRetrievalChain([file_path]).create_chain()
    return pdf

# 노드 정의

# 문서 검색 노드
def retrieve(state):
    print("==== [RETRIEVE] ====")
    question = state["question"]

    # 문서 검색 수행
    documents = pdf_retriever.invoke(question)
    return {"documents": documents}

# 답변 생성 노드
def generate(state):
    # LangChain Hub에서 프롬프트 가져오기(RAG 프롬프트는 자유롭게 수정 가능)
    prompt = hub.pull("teddynote/rag-prompt")
    # LLM 초기화
    llm = ChatOpenAI(model_name=selected_model, temperature=0)
    rag_chain = prompt | llm | StrOutputParser()

    print("==== [GENERATE] ====")
    # 질문과 문서 검색 결과 가져오기
    question = state["question"]
    documents = state["documents"]

    # RAG 답변 생성
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"generation": generation}

# 문서 관련성 평가 노드
def grade_documents(state):
    # LLM 초기화 및 함수 호출을 통한 구조화된 출력 생성
    llm = ChatOpenAI(model=selected_model, temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # 시스템 메시지와 사용자 질문을 포함한 프롬프트 템플릿 생성
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )
    # 문서 검색결과 평가기 생성
    retrieval_grader = grade_prompt | structured_llm_grader
    print("==== [CHECK DOCUMENT RELEVANCE TO QUESTION] ====")
    # 질문과 문서 검색 결과 가져오기
    question = state["question"]
    documents = state["documents"]

    # 각 문서에 대한 관련성 점수 계산
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            # 관련성이 있는 문서 추가
            filtered_docs.append(d)
        else:
            # 관련성이 없는 문서는 건너뛰기
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs}

# 질문 재작성 노드
def transform_query(state):
    # LLM 초기화
    llm = ChatOpenAI(model=selected_model, temperature=0)

    # Query Rewriter 프롬프트 정의(자유롭게 수정이 가능합니다)
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
    for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

    # Query Rewriter 프롬프트 템플릿 생성
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    # Query Rewriter 생성
    question_rewriter = re_write_prompt | llm | StrOutputParser()

    print("==== [TRANSFORM QUERY] ====")
    # 질문과 문서 검색 결과 가져오기
    question = state["question"]
    documents = state["documents"]

    # 질문 재작성
    better_question = question_rewriter.invoke({"question": question})
    return {"question": better_question}

# 웹 검색 노드
def web_search(state):
    web_search_tool = TavilySearch(max_results=3)

    print("==== [WEB SEARCH] ====")
    # 질문과 문서 검색 결과 가져오기
    question = state["question"]

    # 웹 검색 수행
    web_results = web_search_tool.invoke({"query": question})
    web_results_docs = [
        Document(
            page_content=web_result["content"],
            metadata={"source": web_result["url"]},
        )
        for web_result in web_results
    ]
    return {"documents": web_results_docs}

# 질문 라우팅 노드
def route_question(state):
    # LLM 초기화 및 함수 호출을 통한 구조화된 출력 생성
    llm = ChatOpenAI(model=selected_model, temperature=0)
    structured_llm_router = llm.with_structured_output(RouteQuery)

    question = state["question"]

    # 시스템 메시지와 사용자 질문을 포함한 프롬프트 템플릿 생성
    system = """You are an expert at routing a user question to a vectorstore or web search.
        
        The vectorstore contains documents provided by the user (specifically about AI trends, SPRI reports, Samsung Gause, Anthropic, etc).
        
        CRITICAL INSTRUCTION:
        1. If the user asks about "this document", "summary", or specific AI details, ALWAYS choose 'vectorstore'.
        2. Only choose 'web_search' if the question is clearly about unrelated topics (e.g., weather, celebrities) or recent news NOT covered in 2023.
        3. If you are unsure, default to 'vectorstore'."""

    # Routing 을 위한 프롬프트 템플릿 생성
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    
    question_router = route_prompt | structured_llm_router

    print("==== [ROUTE QUESTION] ====")

    # 질문 라우팅
    source = question_router.invoke({"question": question})
    
    # [디버깅용 출력 추가] 실제로 LLM이 뭐라고 생각했는지 확인
    print(f"==== [ROUTER DECISION] Query: {question} -> Source: {source.datasource} ====") 

    # 질문 라우팅 결과에 따른 노드 라우팅
    if source.datasource == "web_search":
        return "web_search"
    elif source.datasource == "vectorstore":
        return "vectorstore"

# 문서 관련성 평가 노드
def decide_to_generate(state):
    print("==== [DECISION TO GENERATE] ====")
    # 문서 검색 결과 가져오기
    filtered_documents = state["documents"]

    if not filtered_documents:
        # 모든 문서가 관련성 없는 경우 질문 재작성
        print(
            "==== [DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY] ===="
        )
        return "transform_query"
    else:
        # 관련성 있는 문서가 있는 경우 답변 생성
        print("==== [DECISION: GENERATE] ====")
        return "generate"
    
def hallucination_check(state):
    
    # 함수 호출을 통한 LLM 초기화
    llm = ChatOpenAI(model=selected_model, temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    # 프롬프트 설정
    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    # 프롬프트 템플릿과 구조화된 LLM 평가기를 결합하여 답변 평가기 생성
    answer_grader = answer_prompt | structured_llm_grader

    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # 프롬프트 설정
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

    # 프롬프트 템플릿 생성
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    # 환각 평가기 생성
    hallucination_grader = hallucination_prompt | structured_llm_grader
    print("==== [CHECK HALLUCINATIONS] ====")
    # 질문과 문서 검색 결과 가져오기
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # 환각 평가
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Hallucination 여부 확인
    if grade == "yes":
        print("==== [DECISION: GENERATION IS GROUNDED IN DOCUMENTS] ====")

        # 답변의 관련성(Relevance) 평가
        print("==== [GRADE GENERATED ANSWER vs QUESTION] ====")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score

        # 관련성 평가 결과에 따른 처리
        if grade == "yes":
            print("==== [DECISION: GENERATED ANSWER ADDRESSES QUESTION] ====")
            return "relevant"
        else:
            print("==== [DECISION: GENERATED ANSWER DOES NOT ADDRESS QUESTION] ====")
            return "not relevant"
    else:
        print("==== [DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY] ====")
        return "hallucination"

# 문서 포맷팅 함수
def format_docs(docs):
    return "\n\n".join(
        [
            f'<document><content>{doc.page_content}</content><source>{doc.metadata["source"]}</source><page>{doc.metadata["page"]+1}</page></document>'
            for doc in docs
        ]
    )


# 메시지 관련 함수
def print_messages():
    """
    저장된 메시지를 화면에 출력하는 함수입니다.
    (데이터 형식이 맞지 않아도 에러가 나지 않도록 예외 처리 추가)
    """
    for role, content_list in st.session_state["messages"]:
        with st.chat_message(role):
            for content in content_list:
                if isinstance(content, list):
                    # [방어 코드] 리스트 길이가 2개인지 확인
                    if len(content) == 2:
                        message_type, message_content = content
                    elif len(content) == 1:
                        # 데이터가 1개만 있으면 텍스트로 간주하고 처리
                        message_type = MessageType.TEXT
                        message_content = content[0]
                    else:
                        # 알 수 없는 형식이면 건너뜀
                        continue

                    # 타입에 따라 다르게 렌더링
                    if message_type == MessageType.TEXT:
                        st.markdown(message_content)
                    elif message_type == MessageType.FIGURE:
                        st.pyplot(message_content)
                    elif message_type == MessageType.CODE:
                        st.code(message_content, language="python")
                    elif message_type == MessageType.DATAFRAME:
                        st.dataframe(message_content)
                elif isinstance(content, str):
                    # 리스트가 아니라 문자열이 바로 들어있는 경우 처리
                    st.markdown(content)


def add_message(role: MessageRole, content: List[Union[MessageType, str]]):
    """
    새로운 메시지를 저장하는 함수입니다.

    Args:
        role (MessageRole): 메시지 역할 (사용자 또는 어시스턴트)
        content (List[Union[MessageType, str]]): 메시지 내용
    """
    messages = st.session_state["messages"]
    if messages and messages[-1][0] == role:
        messages[-1][1].extend([content])  # 같은 역할의 연속된 메시지는 하나로 합칩니다
    else:
        messages.append([role, [content]])  # 새로운 역할의 메시지는 새로 추가합니다


# --- 사이드바 설정 ---
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    # PDF 로직이므로 type을 pdf로 변경
    uploaded_file = st.file_uploader("PDF 파일을 업로드 해주세요.", type=["pdf"])
    # 모델명 수정 (실제 존재하는 모델명으로)
    selected_model = st.selectbox(
        "OpenAI 모델을 선택해주세요.",
        ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0,
    )
    apply_btn = st.button("설정 적용 및 그래프 생성")


# 질문 처리 함수
def ask(query):
    if st.session_state["graph"] is None:
        st.error("먼저 파일을 업로드하고 '설정 적용' 버튼을 눌러주세요.")
        return

    add_message(MessageRole.USER, [MessageType.TEXT, query])
    with st.chat_message("user"):
        st.write(query)

    graph = st.session_state["graph"]

    # thread_id는 대화 동안 고정 권장
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = random_uuid()

    config = RunnableConfig(
        recursion_limit=10,
        configurable={"thread_id": st.session_state["thread_id"]},
    )

    with st.chat_message("assistant"):
            result_state = graph.invoke(
                {"question": query}, 
                config=config
            )

            ai_answer = result_state["generation"]
            st.markdown(ai_answer)

    add_message(MessageRole.ASSISTANT, [MessageType.TEXT, ai_answer])


def build_graph():

    # 그래프 상태 초기화
    workflow = StateGraph(GraphState)

    # 노드 정의
    workflow.add_node("web_search", web_search)  # 웹 검색
    workflow.add_node("retrieve", retrieve)  # 문서 검색
    workflow.add_node("grade_documents", grade_documents)  # 문서 평가
    workflow.add_node("generate", generate)  # 답변 생성
    workflow.add_node("transform_query", transform_query)  # 쿼리 변환

    # 그래프 빌드
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "web_search": "web_search",  # 웹 검색으로 라우팅
            "vectorstore": "retrieve",  # 벡터스토어로 라우팅
        },
    )
    workflow.add_edge("web_search", "generate")  # 웹 검색 후 답변 생성
    workflow.add_edge("retrieve", "grade_documents")  # 문서 검색 후 평가
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",  # 쿼리 변환 필요
            "generate": "generate",  # 답변 생성 가능
        },
    )
    workflow.add_edge("transform_query", "retrieve")  # 쿼리 변환 후 문서 검색
    workflow.add_conditional_edges(
        "generate",
        hallucination_check,
        {
            "hallucination": "generate",  # Hallucination 발생 시 재생성
            "relevant": END,  # 답변의 관련성 여부 통과
            "not relevant": "transform_query",  # 답변의 관련성 여부 통과 실패 시 쿼리 변환
        },
    )

    # 그래프 컴파일
    return workflow.compile(checkpointer=MemorySaver())

# 메인 로직
if clear_btn:
    st.session_state["messages"] = []  # 대화 내용 초기화

if uploaded_file and apply_btn:
    pdf = embed_file(uploaded_file)
    pdf_retriever = pdf.retriever
    pdf_chain = pdf.chain

    st.session_state["graph"] = build_graph()
    st.success("시스템 준비 완료! 질문을 입력하세요.")
elif apply_btn and not uploaded_file:
    st.warning("파일을 업로드 해주세요.")

print_messages()  # 저장된 메시지 출력
user_input = st.chat_input("궁금한 내용을 물어보세요!")  # 사용자 입력 받기
if user_input:
    ask(user_input)  # 사용자 질문 처리
