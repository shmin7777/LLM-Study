## LangChain과 LangGraph의 차이점
* LangChain은 간단한 LLM 어플리케이션이나 RAG시스템 구축하는데 주로 사용, 빠른 개발
* LangGraph는 복잡한 AI 시스템이나 다중 에이전트 시스템을 구축하는데 적합
* LangGraph는 노드와 엣지로 구성된 유연한 워크플로우를 만들수 있음, 여러 단계의 노드를 조건부, 병렬로 처리할 수 있다.
* LangChain 더 알아보기 -> https://hongs-coding.tistory.com/272


## State(상태) -> 노드에서 State를 사용
```py
class BasicState(TypedDict):
    user_input: str
    ai_response: str
    conversation_history: Annotated[list[str], add]
```
* 노드들은 state를 통해 데이터를 주고 받는다.
* 기본스키마(Single Schema, 입력과 출력이 동일)
& 명시적 입출력 스키마(Explicit Input/Output Schema, 입력과 출력 별도로 정의)가 있음
* 다중 스키마 (Multiple Schema, 내부 노드 간 통신을 위한 private 스키마를 포함하는 복잡한 구조)
* Reducer(리듀서) 상태 업데이트 로직을 정의. 각 노드가 반환하는 업데이트를 기존 상태에 어떻게 적용할지 결정함

```
왜 리듀서가 필요한가?

여러 노드가 동시에 또는 순차적으로 실행될 때, 각 노드의 출력을 상태에 통합하는 방법이 필요합니다
단순히 덮어쓰기만 하면 이전 정보가 손실될 수 있습니다
리듀서를 통해 누적, 병합, 최대값 유지 등 다양한 업데이트 전략을 구현할 수 있습니다
```

## Node(노드) -> 실행할 함수의 단위
```py
# node 
def chatbot_node(state: BasicState) -> BasicState: 
    response = f"사용자님이 '{state['user_input']}'라고 하셨군요!"
    return {
        "ai_response": response,
        "conversation_history": [f"User {state['user_input']}", f"AI: {response}"],
    }

# LangGraph 생성
graph = StateGraph(BasicState)
# graph에 node 등록
graph.add_node("chat", chatbot_node)

```
* 현재 state를 입력으로 받아 처리
* 각 노드는 독립적으로 실행
* 여러 노드를 연결하여 복잡한 워크플로우 구성
* 책임: 
1. 데이터 변환 및 처리, 데이터 검증
2. 상태 관리
3. 흐름 제어, 조건부 로직을 통해 다음에 실행될 노드를 결정하거나 특정 조건에서 그래프 실행 종료
* 각 노드별로 필요에 따라 병렬처리나 캐싱을 적용할 수 있다.
* 동기노드, 비동기 노드
* RunnableConfig로 런타임에 노드의 동작을 조정 가능
* 데코레이터 패턴 노드(@wraps(func)) 얘도 Spring의 AOP같은게 있음


## Edge(엣지) -> flow 정의
```py
# langGraph 생성
workflow = StateGraph(WorkflowState)
# step1, step2, step3은 node 함수
# node 등록
workflow.add_node("step1", step1)
workflow.add_node("step2", step2)
workflow.add_node("step3", step3)

workflow.add_edge(START, "step1")    # 1. 시작 -> step1
workflow.add_edge("step1", "step2")  # 2. step1 -> step2
workflow.add_edge("step2", "step3")  # 3. step2 -> step3
workflow.add_edge("step3", END)      # 4. step3 -> 종료
app = graph.compile()
result = app.invoke({"user_input": "안녕하세요 나는 누구에요"})


# 조건부 엣지 (Conditional Edge)
graph.add_conditional_edges(
    "decision_node",
    routing_function,
    {
        "option_a": "node_a",
        "option_b": "node_b"
    }
)

```
* 방향성: 한 노드에서 다른 노드로의 단방향 연결
* 상태 전달: 노드 간 상태 정보 전달
* 흐름 제어: 실행 순서와 조건 결정
* 병렬 실행: 여러 엣지가 동시에 활성화 가능

### 흐름제어
* 조건부 분기를 통해 런타임 상황에 따라 다른 실행 경로를 선택할 수 있게한다.
* 조건부 엣지의 핵심은 라우팅 함수
* 라우팅 함수는 현재 상태를 분석하여 다음에 실행할 노드를 결정하는 로직
```py
from typing import Literal

class ControlFlowState(TypedDict):
    value: int
    path_taken: str
    result: str

def evaluate(state: ControlFlowState) -> dict:
    """값을 평가하는 노드"""
    value = state["value"]

    if value > 100:
        path = "high"
    elif value > 50:
        path = "medium"
    else:
        path = "low"

    return {
        "path_taken": path,
        "result": f"Value {value} is {path}"
    }

def handle_high(state: ControlFlowState) -> dict:
    """높은 값 처리"""
    return {"result": f"HIGH: Special handling for {state['value']}"}

def handle_medium(state: ControlFlowState) -> dict:
    """중간 값 처리"""
    return {"result": f"MEDIUM: Standard handling for {state['value']}"}

def handle_low(state: ControlFlowState) -> dict:
    """낮은 값 처리"""
    return {"result": f"LOW: Basic handling for {state['value']}"}

# 라우팅 함수
def route_by_value(state: ControlFlowState) -> Literal["high", "medium", "low"]:
    """상태에 따라 경로 결정"""
    return state["path_taken"]

# 그래프 구성
control_graph = StateGraph(ControlFlowState)
control_graph.add_node("evaluate", evaluate)
control_graph.add_node("handle_high", handle_high)
control_graph.add_node("handle_medium", handle_medium)
control_graph.add_node("handle_low", handle_low)

# 조건부 엣지로 흐름 제어
control_graph.add_edge(START, "evaluate")
control_graph.add_conditional_edges(
    "evaluate",
    route_by_value,
    {
        "high": "handle_high",
        "medium": "handle_medium",
        "low": "handle_low"
    }
)
control_graph.add_edge("handle_high", END)
control_graph.add_edge("handle_medium", END)
control_graph.add_edge("handle_low", END)
```


참고: https://wikidocs.net/book/16723
